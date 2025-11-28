from __future__ import annotations

import base64
import json
import logging
from typing import AsyncIterator, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from sse_starlette.sse import EventSourceResponse

from agent_graph import AgentState, stream_agent_events

logger = logging.getLogger("backend")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Medical Pipeline API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
async def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/analyze/")
async def analyze_case(
    prompt: str = Form(..., min_length=0, max_length=2000),
    image: Optional[UploadFile] = File(default=None),
) -> EventSourceResponse:
    image_b64: Optional[str] = None

    if image:
        content = await image.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded image is empty.")
        image_b64 = base64.b64encode(content).decode("utf-8")

    async def event_stream() -> AsyncIterator[str]:
        try:
            yield json.dumps({"event": "status", "state": "processing"})

            initial_state: AgentState = {
                "messages": [HumanMessage(content=prompt)],
                "image_data": image_b64,
                "medical_report": None,
                "final_answer": None,
            }

            async for payload in stream_agent_events(initial_state):
                yield json.dumps(payload)
        except Exception as exc:
            logger.exception("Streaming workflow failed.")
            yield json.dumps({"event": "error", "message": str(exc)})

    return EventSourceResponse(event_stream(), media_type="text/event-stream")
