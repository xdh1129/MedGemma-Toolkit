from __future__ import annotations

from typing import AsyncIterator, List, Optional, TypedDict

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ollama_base_url: str = Field(default="http://ollama:11434")
    vlm_model: str = Field(default="hf.co/unsloth/medgemma-27b-it-GGUF:Q4_K_M")
    llm_model: str = Field(default="llama3.1:latest")
    request_timeout_seconds: float = Field(default=180.0, gt=0)


settings = Settings()


class AgentState(TypedDict):
    messages: List[BaseMessage]
    image_data: Optional[str]
    medical_report: Optional[str]
    final_answer: Optional[str]


radiologist_llm = ChatOllama(
    model=settings.vlm_model,
    base_url=settings.ollama_base_url,
    temperature=0,
    timeout=settings.request_timeout_seconds,
)

doctor_llm = ChatOllama(
    model=settings.llm_model,
    base_url=settings.ollama_base_url,
    temperature=0,
    timeout=settings.request_timeout_seconds,
)


def _chunk_text(content: str | List[dict] | None) -> str:
    """Extracts plain text from a streamed chunk."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def _extract_user_prompt(messages: List[BaseMessage]) -> str:
    """Pulls the latest human-authored text from the message history."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                text_parts = [part.get("text") for part in content if isinstance(part, dict) and part.get("type") == "text"]
                return "\n".join([part for part in text_parts if part])
    return ""


def radiologist_node(state: AgentState) -> AgentState:
    """Runs the vision model when an image is present to produce a medical report."""
    image_data = state.get("image_data")
    user_prompt = _extract_user_prompt(state["messages"])

    human_content = [
        {
            "type": "text",
            "text": (
                "You are a radiologist. Analyze the provided medical image in detail. "
                "Focus on clinically relevant findings, note uncertainties, and avoid speculation."
            ),
        }
    ]

    if user_prompt:
        human_content.append({"type": "text", "text": f"Clinician question: {user_prompt}"})

    if image_data:
        human_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}})

    response = radiologist_llm.invoke([HumanMessage(content=human_content)])

    updated_messages = [*state["messages"], response] if isinstance(response, AIMessage) else state["messages"]

    return {**state, "messages": updated_messages, "medical_report": response.content}


def doctor_node(state: AgentState) -> AgentState:
    """Synthesizes the user query and medical report into a final answer."""
    user_prompt = _extract_user_prompt(state["messages"])
    medical_report = state.get("medical_report") or "No imaging report was generated."

    prompt = (
        "You are the attending physician. Review the clinician's question and the medical imaging report "
        "and provide a concise, clinically safe answer. If information is missing, state the limitations.\n\n"
        f"Clinician question:\n{user_prompt}\n\n"
        f"Imaging report:\n{medical_report}\n\n"
        "Respond with clear bullet points when helpful and keep the tone professional."
    )

    response = doctor_llm.invoke(
        [
            SystemMessage(content="You are a careful medical doctor. Avoid overconfident claims."),
            HumanMessage(content=prompt),
        ]
    )

    updated_messages = [*state["messages"], response] if isinstance(response, AIMessage) else state["messages"]

    return {**state, "messages": updated_messages, "final_answer": response.content}


def router(state: AgentState) -> str:
    """Route to the radiologist if an image is present and no report exists; otherwise go to the doctor."""
    if state.get("image_data") and not state.get("medical_report"):
        return "radiologist"
    return "doctor"


graph = StateGraph(AgentState)
graph.add_node("radiologist", radiologist_node)
graph.add_node("doctor", doctor_node)
graph.add_conditional_edges(START, router, {"radiologist": "radiologist", "doctor": "doctor"})
graph.add_edge("radiologist", "doctor")
graph.add_edge("doctor", END)

medical_graph = graph.compile()


# --- Streaming helpers for SSE ---
async def _stream_radiologist(state: AgentState) -> AsyncIterator[dict]:
    """Stream radiologist (VLM) tokens and update state."""
    image_data = state.get("image_data")
    user_prompt = _extract_user_prompt(state["messages"])

    human_content = [
        {
            "type": "text",
            "text": (
                "You are a radiologist. Analyze the provided medical image in detail. "
                "Focus on clinically relevant findings, note uncertainties, and avoid speculation."
            ),
        }
    ]

    if user_prompt:
        human_content.append({"type": "text", "text": f"Clinician question: {user_prompt}"})

    if image_data:
        human_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}})

    report = ""
    async for chunk in radiologist_llm.astream([HumanMessage(content=human_content)]):
        if isinstance(chunk, AIMessageChunk):
            token = _chunk_text(chunk.content)
            if token:
                report += token
                yield {"event": "vlm_token", "token": token}

    state["medical_report"] = report
    state["messages"] = [*state["messages"], AIMessage(content=report)]


async def _stream_doctor(state: AgentState) -> AsyncIterator[dict]:
    """Stream doctor (LLM) tokens and update state."""
    user_prompt = _extract_user_prompt(state["messages"])
    medical_report = state.get("medical_report") or "No imaging report was generated."

    prompt = (
        "You are the attending physician. Review the clinician's question and the medical imaging report "
        "and provide a concise, clinically safe answer. If information is missing, state the limitations.\n\n"
        f"Clinician question:\n{user_prompt}\n\n"
        f"Imaging report:\n{medical_report}\n\n"
        "Respond with clear bullet points when helpful and keep the tone professional."
    )

    answer = ""
    async for chunk in doctor_llm.astream(
        [
            SystemMessage(content="You are a careful medical doctor. Avoid overconfident claims."),
            HumanMessage(content=prompt),
        ]
    ):
        if isinstance(chunk, AIMessageChunk):
            token = _chunk_text(chunk.content)
            if token:
                answer += token
                yield {"event": "llm_token", "token": token}

    state["final_answer"] = answer
    state["messages"] = [*state["messages"], AIMessage(content=answer)]


async def stream_agent_events(initial_state: AgentState) -> AsyncIterator[dict]:
    """
    Drive the agentic workflow with streaming tokens for SSE.

    Yields event payloads compatible with the existing frontend stream handler.
    """
    state = {**initial_state}

    if state.get("image_data") and not state.get("medical_report"):
        async for payload in _stream_radiologist(state):
            yield payload

    async for payload in _stream_doctor(state):
        yield payload

    yield {"event": "done", "vlm_output": state.get("medical_report"), "llm_report": state.get("final_answer")}
