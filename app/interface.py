"""Gradio interface wiring for SafeSOS.

This file defines UI components, callbacks, and small HTML helpers.
Main styling is in styles.py.
"""

from __future__ import annotations

import gradio as gr

from app.inference import GesturePredictor
from app.styles import CUSTOM_CSS, STATUS_COLOR_MAP


# Predictor singleton

_PREDICTOR: GesturePredictor | None = None


def get_predictor() -> GesturePredictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = GesturePredictor()
    return _PREDICTOR


# JS callbacks

# Browser TTS callback.
TTS_JS = r"""
async (message, status) => {
    if (!message || status === "IDLE") return [message, status];
    if (!("speechSynthesis" in window)) return [message, status];
    try {
        window.speechSynthesis.cancel();
        const utter = new SpeechSynthesisUtterance(message);
        utter.rate = status === "ABSTAINED" ? 0.92 : 1.02;
        utter.pitch = status === "ABSTAINED" ? 0.85 : 1.05;
        utter.volume = 1.0;
        window.speechSynthesis.speak(utter);
    } catch (e) { console.error(e); }
    return [message, status];
}
"""


# Inference callbacks

def predict_frame(image):
    predictor = get_predictor()
    result, probs = predictor.predict_with_full_probs(image)
    color = STATUS_COLOR_MAP.get(result.status, "#888")
    top5 = dict(sorted(probs.items(), key=lambda kv: -kv[1])[:5])
    return (
        result.message,
        result.status,
        result.confidence,
        top5,
        _status_pill(result.status, color, result.confidence),
        _telemetry_line(result.status, result.confidence),
    )


def manual_predict(image):
    return predict_frame(image)


# Interface construction

def build_interface() -> gr.Blocks:
    with gr.Blocks(
        css=CUSTOM_CSS,
        theme=gr.themes.Base(),
        title="SafeSOS // Silent Emergency Communication",
        analytics_enabled=False,
    ) as demo:
        # Background laser layer.
        gr.HTML(_laser_svg())

        # Header.
        gr.HTML(_header_html())

        # Main layout.
        with gr.Row(elem_classes="console-row"):
            # Left panel: webcam + telemetry.
            with gr.Column(scale=5, elem_classes="left-pane"):
                gr.HTML('<div class="frame-label"><span class="dot-live"></span>LIVE FEED &middot; <span class="mono">CAM-01</span> &middot; <span class="hint">CLICK ⏺ TO START STREAMING</span></div>')
                with gr.Group(elem_classes="viewfinder"):
                    camera = gr.Image(
                        sources=["webcam"],
                        streaming=True,
                        type="numpy",
                        label="",
                        elem_classes="camera-feed",
                        show_label=False,
                        height=420,
                    )
                # Telemetry strip.
                telemetry_html = gr.HTML(
                    value=_telemetry_line("IDLE", 0.0),
                    elem_id="telemetry-line",
                )

            # Right panel: status + prediction outputs.
            with gr.Column(scale=5, elem_classes="right-pane"):
                gr.HTML('<div class="frame-label"><span class="dot-pulse"></span>INFERENCE STATE</div>')

                # Status widget.
                status_html = gr.HTML(
                    value=_status_pill("IDLE", "#6c7a89", 0.0),
                    elem_id="status-container",
                )

                # Message card.
                message_html = gr.HTML(
                    value=_message_card("Awaiting input signal.", "IDLE"),
                    elem_id="message-card-container",
                )

                # Hidden state for JS TTS.
                spoken_message = gr.Textbox(value="", visible=False)
                spoken_status = gr.Textbox(value="IDLE", visible=False)

                # Confidence and top predictions.
                gr.HTML('<div class="frame-label" style="margin-top:1.25rem;"><span class="dot-pulse"></span>CONFIDENCE</div>')
                confidence_bar = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.0,
                    step=0.001,
                    label="",
                    interactive=False,
                    elem_classes="confidence-slider",
                    show_label=False,
                )

                with gr.Accordion("◢ TOP PREDICTIONS", open=False, elem_classes="top-accordion"):
                    probs_label = gr.Label(
                        label="",
                        num_top_classes=5,
                        show_label=False,
                    )

        # About and footer.
        with gr.Accordion("◢ SYSTEM INFORMATION", open=False, elem_classes="about-panel"):
            gr.Markdown(_about_md())

        gr.HTML(_footer_html())

        # Streaming wiring.
        outputs = [
            spoken_message,
            spoken_status,
            confidence_bar,
            probs_label,
            status_html,
            telemetry_html,
        ]
        camera.stream(
            fn=predict_frame,
            inputs=camera,
            outputs=outputs,
            stream_every=0.3,
            show_progress="hidden",
        )

        # Trigger TTS when message changes.
        spoken_message.change(
            fn=None,
            inputs=[spoken_message, spoken_status],
            outputs=[spoken_message, spoken_status],
            js=TTS_JS,
        )

        # Keep message card in sync.
        spoken_message.change(
            fn=lambda m, s: _message_card(m, s),
            inputs=[spoken_message, spoken_status],
            outputs=message_html,
        )

    return demo


# HTML helpers

def _header_html() -> str:
    return """
    <div class="hud-header">
        <div class="hud-left">
            <div class="brand-row">
                <span class="brand-mark">◣◢</span>
                <span class="brand-name">SafeSOS</span>
                <span class="brand-version">v0.1 / TWELVE-EPOCH BUILD</span>
            </div>
            <div class="tagline">SILENT &middot; EMERGENCY &middot; GESTURE PROTOCOL</div>
        </div>
        <div class="hud-right">
            <div class="hud-tag tag-red">SELECTIVE&nbsp;PREDICTION</div>
            <div class="hud-tag tag-grey">MOBILENETV3&nbsp;·&nbsp;T=1.10</div>
        </div>
    </div>
    """


def _laser_svg() -> str:
    """Background laser SVG layer with mild turbulence animation."""
    return """
    <div class="laser-stage">
      <svg viewBox="0 0 1200 700" preserveAspectRatio="xMidYMid slice">
        <defs>
          <!-- Turbulence/displacement filter -->
          <filter id="dust" x="-30%" y="-30%" width="160%" height="160%">
            <feTurbulence type="fractalNoise"
                          baseFrequency="0.006 0.04"
                          numOctaves="2" seed="3">
              <animate attributeName="seed"
                       values="3;43;83;43;3"
                       dur="14s" repeatCount="indefinite"/>
            </feTurbulence>
            <feDisplacementMap in="SourceGraphic" scale="42"
                               xChannelSelector="R" yChannelSelector="G"/>
            <feGaussianBlur stdDeviation="0.6"/>
          </filter>

          <!-- Beam gradients -->
          <linearGradient id="beamGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%"   stop-color="#2ecc71" stop-opacity="0"/>
            <stop offset="12%"  stop-color="#2ecc71" stop-opacity="0.35"/>
            <stop offset="45%"  stop-color="#aef5c8" stop-opacity="0.95"/>
            <stop offset="55%"  stop-color="#ffffff" stop-opacity="1"/>
            <stop offset="65%"  stop-color="#aef5c8" stop-opacity="0.95"/>
            <stop offset="88%"  stop-color="#2ecc71" stop-opacity="0.35"/>
            <stop offset="100%" stop-color="#2ecc71" stop-opacity="0"/>
          </linearGradient>

          <linearGradient id="beamGradDim" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%"   stop-color="#2ecc71" stop-opacity="0"/>
            <stop offset="20%"  stop-color="#2ecc71" stop-opacity="0.25"/>
            <stop offset="50%"  stop-color="#8df0b3" stop-opacity="0.75"/>
            <stop offset="80%"  stop-color="#2ecc71" stop-opacity="0.25"/>
            <stop offset="100%" stop-color="#2ecc71" stop-opacity="0"/>
          </linearGradient>
        </defs>

        <g filter="url(#dust)">
          <!-- Beam A -->
          <rect class="beam-a" x="-80" y="220" width="1400" height="16"
                fill="url(#beamGrad)"/>

          <!-- Beam B -->
          <rect class="beam-b" x="-80" y="380" width="1400" height="11"
                fill="url(#beamGradDim)"/>

          <!-- Beam C -->
          <rect class="beam-c" x="-80" y="510" width="1400" height="7"
                fill="url(#beamGradDim)"/>
        </g>
      </svg>
    </div>
    """


def _status_pill(status: str, color: str, confidence: float) -> str:
    intensity = min(1.0, max(0.0, confidence))
    return f"""
    <div class="status-shell" style="--accent:{color}">
        <div class="status-ring">
            <svg viewBox="0 0 100 100" class="ring-svg">
                <circle cx="50" cy="50" r="44" class="ring-track"/>
                <circle cx="50" cy="50" r="44" class="ring-fill"
                        style="stroke-dashoffset:{276 * (1 - intensity):.1f};"/>
            </svg>
            <div class="status-text">
                <div class="status-label">{status}</div>
                <div class="status-conf">{intensity * 100:.1f}%</div>
            </div>
        </div>
    </div>
    """


def _message_card(message: str, status: str) -> str:
    color = STATUS_COLOR_MAP.get(status, "#888")
    safe_msg = (message or "—").replace('"', '&quot;')
    return f"""
    <div class="message-card" style="--accent:{color}">
        <div class="corner tl"></div>
        <div class="corner tr"></div>
        <div class="corner bl"></div>
        <div class="corner br"></div>
        <div class="message-text">{safe_msg}</div>
        <div class="message-meta">{status}</div>
    </div>
    """


def _telemetry_line(status: str, confidence: float) -> str:
    """Render telemetry strip below the camera."""
    bars = max(1, min(10, int(confidence * 10)))
    bar_str = "▮" * bars + "▯" * (10 - bars)
    return f"""
    <div class="telemetry">
        <span class="t-key">SIG</span>
        <span class="t-bar">{bar_str}</span>
        <span class="t-key">ST</span>
        <span class="t-val">{status}</span>
        <span class="t-key">CONF</span>
        <span class="t-val">{confidence:.3f}</span>
        <span class="t-key">FPS</span>
        <span class="t-val">~3.0</span>
    </div>
    """


def _footer_html() -> str:
    return """
    <div class="hud-footer">
        <span>RESEARCH PROTOTYPE / NOT A MEDICAL DEVICE</span>
        <span class="footer-sep">//</span>
        <span>BUILT FOR THE CV MODULE PROJECT</span>
    </div>
    """


def _about_md() -> str:
    return """
**SafeSOS** is a research prototype that recognises a small vocabulary
of static hand gestures and translates them into spoken emergency
messages — designed for high-stakes silent-communication scenarios:
non-verbal ICU patients, hearing-impaired users in distress, victims
of abuse using covert distress signals.

**Vocabulary** (10 classes including a no-gesture abstention anchor):

| Gesture | Spoken message |
|---|---|
| 📞 call | "I need to make a call. Please help." |
| ✋ palm | "I need help." |
| ✋ stop | "STOP. Do not approach." |
| ✊ fist | "ALERT. I am in danger." |
| 👌 ok | "I am OK." |
| ✌️ peace | "Confirmed. I am safe." |
| ☝️ one / ✌️ two / 🤟 three | numeric (person count, floor) |
| (no gesture) | system stays silent |

**Selective prediction.** When model confidence falls below τ = 0.75
the system abstains — "I'm not sure. Please try again." — rather than
guessing. In high-stakes contexts, abstaining is the correct response
to uncertainty.

**Pipeline.** Webcam → 224×224 → MobileNetV3-Small (10-class, T=1.10) →
softmax → threshold gate → spoken message via the browser's Web
Speech API.

This is a research prototype. It is **not** a medical device and
**must not** be relied on for actual emergencies.
"""
