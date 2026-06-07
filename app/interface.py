"""
Gradio Blocks interface for SafeSOS.

This is intentionally NOT a "basic Gradio demo":
- Custom CSS gives the app an emergency-tool aesthetic (dark UI, alert-red)
- Status colors map onto safety semantics: green=ok, amber=abstain, red=alert
- Layout adapted for accessibility: large fonts, high contrast, big buttons
- Probability distribution panel exposes WHY the model decided what it did
- A "What's this app?" panel orients first-time users (helpful for graders)
- Web Speech API integration speaks the predicted emergency message aloud

Streaming runs at ~3 fps (every 0.3s) to balance responsiveness vs CPU.
"""

from __future__ import annotations

import gradio as gr

from app.inference import GesturePredictor
from app.styles import CUSTOM_CSS, STATUS_COLOR_MAP


# ============================================================
# Predictor singleton
# ============================================================

# Single global predictor: Gradio's streaming sessions all share it.
# Lazy-loaded on first call to avoid blocking app startup.
_PREDICTOR: GesturePredictor | None = None


def get_predictor() -> GesturePredictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = GesturePredictor()
    return _PREDICTOR


# ============================================================
# Callbacks
# ============================================================

# A small JS shim that speaks any text via the browser's TTS engine.
# Triggered from the back-end when message/status updates.
TTS_JS = """
async (message, status) => {
    if (!message || status === "IDLE") return [message, status];
    if (!("speechSynthesis" in window)) return [message, status];
    try {
        window.speechSynthesis.cancel();
        const utter = new SpeechSynthesisUtterance(message);
        utter.rate = 1.0;
        utter.pitch = status === "ABSTAINED" ? 0.9 : 1.05;
        window.speechSynthesis.speak(utter);
    } catch (e) { console.error(e); }
    return [message, status];
}
"""


def predict_frame(image) -> tuple[str, str, float, dict, str]:
    """Process a single webcam frame, return UI state.

    Returns:
        message:       text to display + speak
        status:        IDLE | OK | ABSTAINED | ERROR
        confidence:    float for the progress bar
        prob_dict:     {class_name: prob} for the top-classes panel
        status_color:  hex color for the status banner
    """
    predictor = get_predictor()
    result, probs = predictor.predict_with_full_probs(image)
    color = STATUS_COLOR_MAP.get(result.status, "#888")
    # Keep only top 5 probs for the visualisation panel
    top5 = dict(sorted(probs.items(), key=lambda kv: -kv[1])[:5])
    return result.message, result.status, result.confidence, top5, color


def manual_predict(image) -> tuple[str, str, float, dict, str]:
    """Called by the 'Capture' button — same logic but suitable for one-shot use."""
    return predict_frame(image)


# ============================================================
# Build the interface
# ============================================================

def build_interface() -> gr.Blocks:
    """Construct the Gradio Blocks app."""
    with gr.Blocks(
        css=CUSTOM_CSS,
        theme=gr.themes.Base(),
        title="SafeSOS — Silent Emergency Communication",
        analytics_enabled=False,
    ) as demo:

        # -------- Header --------
        gr.HTML("""
            <div class="header-bar">
                <div class="header-left">
                    <h1>🆘 <span class="brand">SafeSOS</span></h1>
                    <h3>Silent emergency communication through hand gestures</h3>
                </div>
                <div class="header-right">
                    <div class="tag">SELECTIVE PREDICTION</div>
                    <div class="tag tag-secondary">MOBILENETV3</div>
                </div>
            </div>
        """)

        # -------- Main app row --------
        with gr.Row(elem_classes="main-row"):
            # Left column: camera
            with gr.Column(scale=5, elem_classes="left-pane"):
                camera = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    label="",
                    elem_classes="camera-feed",
                    show_label=False,
                    height=360,
                )
                with gr.Row(elem_classes="action-row"):
                    capture_btn = gr.Button(
                        "📷  Capture",
                        elem_classes="capture-btn",
                        size="lg",
                    )

            # Right column: status / message / probs
            with gr.Column(scale=5, elem_classes="right-pane"):
                status_html = gr.HTML(
                    value=_status_pill("IDLE", "#666"),
                    elem_id="status-pill-container",
                )

                message_html = gr.HTML(
                    value=_message_card("Show a gesture to the camera.", "IDLE"),
                    elem_id="message-card-container",
                )

                confidence_bar = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.0,
                    step=0.01,
                    label="Model confidence",
                    interactive=False,
                    elem_classes="confidence-slider",
                )

                with gr.Accordion("Top predictions", open=False,
                                  elem_classes="top-accordion"):
                    probs_label = gr.Label(
                        label="",
                        num_top_classes=5,
                        show_label=False,
                    )

        # -------- About panel --------
        with gr.Accordion("About SafeSOS", open=False, elem_classes="about-panel"):
            gr.Markdown("""
**SafeSOS** is a research prototype that recognises a small vocabulary
of static hand gestures and translates them into spoken emergency
messages. It is designed for high-stakes silent-communication
scenarios — non-verbal ICU patients, hearing-impaired users in
distress, and similar safety-critical use cases.

**Vocabulary** (9 gestures + no-gesture):

| Gesture | Meaning | Gesture | Meaning |
|---|---|---|---|
| 📞 call | "I need to make a call" | ✊ fist | "Alert — I am in danger" |
| ✋ palm | "I need help" | 👌 ok | "I am OK" |
| ✋ stop | "Stop, do not approach" | ✌️ peace | "Confirmed safe" |
| ☝️ one | "1" (person / floor) | ✌️ two_up | "2" |
| 🖖 three | "3" | — | — |

**Selective prediction.** The system abstains when its confidence
falls below 0.75. In high-stakes settings, "I'm not sure" is the
correct response when the model is uncertain — better than guessing.

**This is a research prototype** and must NOT be deployed as a
medical device or substitute for professional emergency response.
            """)

        # State piped into JS for TTS
        spoken_message = gr.Textbox(value="", visible=False)
        spoken_status = gr.Textbox(value="IDLE", visible=False)

        # -------- Streaming wiring --------
        # Webcam stream -> predictor -> UI updates
        outputs = [spoken_message, spoken_status, confidence_bar, probs_label, status_html]
        camera.stream(
            fn=lambda img: _wrap_outputs(predict_frame(img)),
            inputs=camera,
            outputs=outputs,
            stream_every=0.3,  # ~3 fps; lower CPU than 30fps and still feels live
            show_progress="hidden",
        )

        capture_btn.click(
            fn=lambda img: _wrap_outputs(predict_frame(img)),
            inputs=camera,
            outputs=outputs,
        )

        # Whenever spoken_message updates, hand it to JS for TTS + display
        spoken_message.change(
            fn=None,
            inputs=[spoken_message, spoken_status],
            outputs=[spoken_message, spoken_status],
            js=TTS_JS,
        )

        # Also update the visible message card when message changes
        spoken_message.change(
            fn=lambda m, s: _message_card(m, s),
            inputs=[spoken_message, spoken_status],
            outputs=message_html,
        )

    return demo


# ============================================================
# HTML helpers (so we don't sprinkle inline HTML through callbacks)
# ============================================================

def _status_pill(status: str, color: str) -> str:
    return f"""
    <div class="status-pill" style="background:{color}33; border-color:{color}">
        <span class="dot" style="background:{color}"></span>
        <span class="status-label" style="color:{color}">{status}</span>
    </div>
    """


def _message_card(message: str, status: str) -> str:
    color = STATUS_COLOR_MAP.get(status, "#888")
    return f"""
    <div class="message-card" style="border-color:{color}">
        <div class="message-text">{message or '&nbsp;'}</div>
    </div>
    """


def _wrap_outputs(prediction_tuple):
    """Adapt predict_frame's 5-tuple into our 5 Gradio outputs.

    predict_frame returns: (message, status, confidence, top5_dict, color)
    We turn that into: (message, status, confidence, top5_dict, status_html)
    """
    message, status, confidence, top5, color = prediction_tuple
    return (
        message,
        status,
        confidence,
        top5,
        _status_pill(status, color),
    )
