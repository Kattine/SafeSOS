"""
Custom CSS for SafeSOS Gradio app.

This is what differentiates the app from a "basic Gradio demo":
a dark, high-contrast, emergency-tool aesthetic with status semantics
encoded in colour.
"""

# Status -> hex colour mapping; reused in both CSS and Python.
STATUS_COLOR_MAP: dict[str, str] = {
    "IDLE":      "#6c7a89",   # neutral grey
    "OK":        "#2ecc71",   # green: confident safe prediction
    "ABSTAINED": "#f39c12",   # amber: model abstained
    "ERROR":     "#e74c3c",   # red: pipeline error
}


CUSTOM_CSS = """
/* ===== Base reset and typography ===== */
:root {
    --bg-deep:    #0a0e1a;
    --bg-card:   #131826;
    --bg-card-2: #1a1f2e;
    --border:    #2c3e50;
    --text:      #e8eaed;
    --text-dim:  #95a5b6;
    --text-mute: #6c7a89;
    --alert-red: #ff4757;
}

.gradio-container {
    background: var(--bg-deep) !important;
    color: var(--text) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Inter",
                 "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
    max-width: 1280px !important;
    margin: 0 auto !important;
    padding: 1.5rem !important;
}

/* Gradio sometimes injects a white footer; hide it */
footer { display: none !important; }

/* ===== Header bar ===== */
.header-bar {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    border-bottom: 1px solid var(--border);
    padding-bottom: 1rem;
    margin-bottom: 1.5rem;
}
.header-bar h1 {
    color: var(--text) !important;
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
    margin: 0 !important;
}
.header-bar .brand {
    color: var(--alert-red) !important;
}
.header-bar h3 {
    color: var(--text-dim) !important;
    font-weight: 400 !important;
    margin: 0.2rem 0 0 0 !important;
    font-size: 1rem !important;
}
.header-right { display: flex; gap: 0.5rem; }
.tag {
    background: rgba(255, 71, 87, 0.12);
    color: var(--alert-red);
    border: 1px solid rgba(255, 71, 87, 0.4);
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.tag-secondary {
    background: rgba(108, 122, 137, 0.15);
    color: var(--text-dim);
    border-color: rgba(108, 122, 137, 0.4);
}

/* ===== Main row ===== */
.main-row { gap: 1.5rem !important; }

/* ===== Camera ===== */
.camera-feed {
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    overflow: hidden;
    background: #000 !important;
}
.camera-feed img, .camera-feed video {
    border-radius: 12px !important;
    width: 100% !important;
    object-fit: cover;
}

.action-row { justify-content: center; margin-top: 0.75rem; }
.capture-btn button {
    background: var(--alert-red) !important;
    color: white !important;
    border: none !important;
    padding: 0.75rem 2rem !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em;
    border-radius: 999px !important;
    text-transform: uppercase;
    cursor: pointer;
    transition: transform 0.1s ease, box-shadow 0.2s ease;
    box-shadow: 0 4px 12px rgba(255, 71, 87, 0.3);
}
.capture-btn button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(255, 71, 87, 0.4);
}
.capture-btn button:active {
    transform: translateY(0);
}

/* ===== Status pill ===== */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 0.9rem;
    border: 2px solid;
    border-radius: 999px;
    font-weight: 700;
    letter-spacing: 0.08em;
    font-size: 0.85rem;
    margin-bottom: 0.75rem;
}
.status-pill .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    animation: pulse 1.4s ease-in-out infinite;
}
.status-pill .status-label {
    font-weight: 700;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.4; }
}

/* ===== Message card ===== */
.message-card {
    background: var(--bg-card);
    border: 2px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem 1.25rem;
    min-height: 110px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: border-color 0.3s ease;
}
.message-text {
    color: var(--text);
    font-size: 1.5rem;
    font-weight: 700;
    line-height: 1.3;
    text-align: center;
}

/* ===== Confidence slider ===== */
.confidence-slider {
    margin-top: 1rem;
}
.confidence-slider label {
    color: var(--text-dim) !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
}

/* ===== Accordion panels ===== */
.top-accordion, .about-panel {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    margin-top: 1rem;
}
.top-accordion summary, .about-panel summary {
    color: var(--text-dim) !important;
    font-weight: 600 !important;
}

/* Label component (top-5 probs) */
.gr-label, .label-wrap {
    background: transparent !important;
}

/* ===== Generic Gradio overrides ===== */
.gr-box, .gr-form, .gr-panel {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
.block.padded { background: transparent !important; }
label.svelte-1gfkn6j { color: var(--text-dim) !important; }

/* Accessibility: focus rings */
button:focus-visible, [role="button"]:focus-visible {
    outline: 3px solid var(--alert-red) !important;
    outline-offset: 2px !important;
}

/* Responsive: stack on narrow screens */
@media (max-width: 800px) {
    .header-bar h1 { font-size: 1.8rem !important; }
    .main-row { flex-direction: column !important; }
    .message-text { font-size: 1.2rem; }
}
"""
