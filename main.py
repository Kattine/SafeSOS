"""
Main entry point for the SafeSOS application.
Launches the Gradio Blocks interface for emergency gesture recognition.

Usage:
    python main.py
"""

from app.interface import build_interface


def main() -> None:
    """Build and launch the Gradio app."""
    demo = build_interface()
    # 127.0.0.1 (instead of 0.0.0.0) keeps the URL bar showing localhost,
    # which is the only HTTP origin browsers will grant camera permission
    # to without HTTPS. On HF Spaces, gradio handles HTTPS itself.
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,     # surface tracebacks in the UI for easier debugging
    )


if __name__ == "__main__":
    main()
