"""
Main entry point for the Silent-SOS application.
Launches the Gradio Blocks interface for emergency gesture recognition.

Usage:
    python main.py
"""

from app.interface import build_interface


def main() -> None:
    """Build and launch the Gradio app."""
    demo = build_interface()
    # share=False locally; HF Spaces handles public hosting automatically
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
