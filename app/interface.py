"""
Gradio Blocks interface for Silent-SOS.

Heavy custom CSS makes this *not* a "basic Gradio app":
- Emergency red/amber/green status colors
- High-contrast, accessibility-first typography
- Real-time webcam stream with prediction overlay
- Selective prediction: clearly shows "ABSTAINED" when confidence is low
"""

import gradio as gr
from app.inference import GesturePredictor
from app.styles import CUSTOM_CSS


# Module-level predictor instance (loaded once at app start)
PREDICTOR: GesturePredictor | None = None


def get_predictor() -> GesturePredictor:
    """Lazy-load the predictor on first call."""
    global PREDICTOR
    if PREDICTOR is None:
        PREDICTOR = GesturePredictor()
    return PREDICTOR


def predict_frame(image) -> tuple[str, str, float]:
    """Run a single frame through the predictor and format the UI response.

    Returns:
        message: the spoken emergency message (or "Waiting...")
        status: one of "OK", "ABSTAINED", "ERROR"
        confidence: model confidence on [0, 1]
    """
    if image is None:
        return "Waiting for camera...", "IDLE", 0.0
    result = get_predictor().predict(image)
    return result.message, result.status, result.confidence


def build_interface() -> gr.Blocks:
    """Construct the Gradio Blocks app."""
    with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Base(), title="Silent-SOS") as demo:
        gr.Markdown(
            """
            # 🆘 Silent-SOS
            ### Silent emergency communication through hand gestures.
            """,
            elem_classes="header",
        )

        with gr.Row():
            with gr.Column(scale=1):
                camera = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    label="Camera",
                    elem_classes="camera-feed",
                )
            with gr.Column(scale=1):
                status_box = gr.Textbox(
                    label="STATUS",
                    value="IDLE",
                    elem_classes="status-box",
                )
                message_box = gr.Textbox(
                    label="MESSAGE",
                    value="Show a gesture to the camera.",
                    elem_classes="message-box",
                )
                confidence_bar = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    label="Model Confidence",
                    interactive=False,
                )

        camera.stream(
            fn=predict_frame,
            inputs=camera,
            outputs=[message_box, status_box, confidence_bar],
            stream_every=0.3,  # 3 fps to avoid overload
        )

        gr.Markdown(
            "Built for the CV module project. "
            "Selective prediction: the model abstains when uncertain.",
            elem_classes="footer",
        )

    return demo
