"""Entry point for launching the SafeSOS Gradio app."""

import os

from app.interface import build_interface


def main() -> None:
    """Build and launch the Gradio app."""
    demo = build_interface()
    # Bind to 0.0.0.0 on Spaces, localhost in local runs.
    on_hf_space = "SPACE_ID" in os.environ
    server_name = "0.0.0.0" if on_hf_space else "127.0.0.1"

    demo.launch(
        server_name=server_name,
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
