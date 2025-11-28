from __future__ import annotations

from pathlib import Path

from agent_graph import medical_graph


def main() -> None:
    graph = medical_graph.get_graph()

    mermaid_text = graph.draw_mermaid()
    mermaid_path = Path(__file__).parent / "medical_graph.mmd"
    mermaid_path.write_text(mermaid_text, encoding="utf-8")
    print(f"✅ Mermaid diagram written to: {mermaid_path}")

    png_path = Path(__file__).parent / "medical_graph.png"
    try:
        png_bytes = graph.draw_mermaid_png()
        png_path.write_bytes(png_bytes)
        print(f"✅ PNG diagram written to: {png_path}")
    except Exception as exc:  # pragma: no cover - best-effort rendering
        print("⚠️  Mermaid PNG render failed (requires optional rendering deps).")
        print(f"Reason: {exc}")


if __name__ == "__main__":
    main()
