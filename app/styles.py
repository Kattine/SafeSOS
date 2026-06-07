"""Custom CSS for the SafeSOS Gradio interface."""

STATUS_COLOR_MAP: dict[str, str] = {
    "IDLE":      "#6c7a89",
    "OK":        "#2ecc71",
    "ABSTAINED": "#f39c12",
    "ERROR":     "#e74c3c",
}


CUSTOM_CSS = r"""
/* Root tokens */
:root {
    /* Neutral grayscale palette */
    --bg-0:       #1a1a1c;
    --bg-1:       #232325;
    --bg-2:       #2a2b2e;
    --bg-card:    #1f2022;
    --bg-card-2:  #2a2b2e;
    --border:     #34363b;
    --border-hi:  #4a4d54;
    --text:       #e9ebef;
    --text-dim:   #a8acb5;
    --text-mute:  #6f747e;
    --accent:     #2ecc71;
    --alert:      #ff5566;
    --warning:    #f5a623;
    --grid:       rgba(180, 185, 200, 0.025);
    --mono: "JetBrains Mono", "SF Mono", "Menlo", "Consolas", monospace;
}

/* Global canvas */
.gradio-container {
    background:
        linear-gradient(180deg,
                        #1c1c20 0%,
                        #2c2c32 50%,
                        #1c1c20 100%) !important;
    color: var(--text) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Inter",
                 "Segoe UI", Roboto, sans-serif;
    max-width: 1300px !important;
    margin: 0 auto !important;
    padding: 1.25rem 1.5rem !important;
    position: relative;
    overflow: hidden;
    min-height: 100vh;
}

/* Rotating spotlight layer */
.gradio-container::before {
    content: "";
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background:
        radial-gradient(circle at 25% 30%,
                        rgba(240, 245, 255, 0.75) 0%,
                        rgba(220, 228, 245, 0.35) 15%,
                        rgba(180, 190, 210, 0.10) 30%,
                        transparent 50%),
        radial-gradient(circle at 75% 70%,
                        rgba(230, 238, 252, 0.65) 0%,
                        rgba(200, 210, 230, 0.28) 18%,
                        rgba(160, 170, 195, 0.08) 32%,
                        transparent 52%),
        radial-gradient(circle at 50% 50%,
                        rgba(250, 252, 255, 0.20) 0%,
                        transparent 35%);
    filter: blur(35px);
    animation: silver-flow 15s linear infinite;
    pointer-events: none;
    z-index: 0;
}

@keyframes silver-flow {
    0%   { transform: rotate(0deg)   scale(1.0); }
    25%  { transform: rotate(90deg)  scale(1.15); }
    50%  { transform: rotate(180deg) scale(1.0); }
    75%  { transform: rotate(270deg) scale(0.88); }
    100% { transform: rotate(360deg) scale(1.0); }
}

/* Subtle grid layer */
.gradio-container::after {
    content: "";
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(var(--grid) 1px, transparent 1px),
        linear-gradient(90deg, var(--grid) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
    z-index: 0;
    opacity: 0.7;
}

/* Green laser SVG layer */
.laser-stage {
    position: absolute;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
    mix-blend-mode: screen;
}
.laser-stage svg {
    width: 100%;
    height: 100%;
    display: block;
}
/* Beam motions */
.beam-a {
    transform-origin: 50% 50%;
    animation: beam-a-pan 19s ease-in-out infinite alternate;
    opacity: 0.85;
}
.beam-b {
    transform-origin: 50% 50%;
    animation: beam-b-pan 26s ease-in-out infinite alternate;
    opacity: 0.55;
}
.beam-c {
    transform-origin: 50% 50%;
    animation: beam-c-pan 34s ease-in-out infinite alternate;
    opacity: 0.4;
}
@keyframes beam-a-pan {
    0%   { transform: rotate(-18deg) translateY(-30px); }
    50%  { transform: rotate(6deg)   translateY(40px);  }
    100% { transform: rotate(20deg)  translateY(-10px); }
}
@keyframes beam-b-pan {
    0%   { transform: rotate(14deg)  translateY(60px);  }
    50%  { transform: rotate(-4deg)  translateY(-20px); }
    100% { transform: rotate(-22deg) translateY(30px);  }
}
@keyframes beam-c-pan {
    0%   { transform: rotate(-8deg)  translateY(-50px); }
    50%  { transform: rotate(10deg)  translateY(10px);  }
    100% { transform: rotate(2deg)   translateY(80px);  }
}

footer { display: none !important; }
.gradio-container > * { position: relative; z-index: 1; }

/* Header */
.hud-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    border-bottom: 1px solid var(--border-hi);
    padding-bottom: 0.85rem;
    margin-bottom: 1.5rem;
}
.brand-row {
    display: flex; align-items: baseline; gap: 0.65rem;
}
.brand-mark {
    color: var(--alert);
    font-family: var(--mono);
    font-size: 1.25rem;
    letter-spacing: -0.05em;
    animation: brand-glow 3s ease-in-out infinite;
}
@keyframes brand-glow {
    0%, 100% { opacity: 0.8; text-shadow: 0 0 6px rgba(255, 71, 87, 0.4); }
    50%      { opacity: 1.0; text-shadow: 0 0 14px rgba(255, 71, 87, 0.9); }
}
.brand-name {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: var(--text);
}
.brand-version {
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--text-mute);
    letter-spacing: 0.08em;
    padding-left: 0.5rem;
    border-left: 1px solid var(--border);
}
.tagline {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--text-dim);
    margin-top: 0.35rem;
    letter-spacing: 0.18em;
}
.hud-right { display: flex; gap: 0.5rem; }
.hud-tag {
    padding: 0.3rem 0.7rem;
    border-radius: 4px;
    font-family: var(--mono);
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    border: 1px solid;
}
.tag-red  {
    background: rgba(255, 71, 87, 0.1);
    color: var(--alert);
    border-color: rgba(255, 71, 87, 0.4);
}
.tag-grey {
    background: rgba(108, 122, 137, 0.1);
    color: var(--text-dim);
    border-color: var(--border-hi);
}

/* Console labels */
.frame-label {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--text-dim);
    letter-spacing: 0.12em;
    margin-bottom: 0.5rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.dot-live, .dot-pulse {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--alert);
    box-shadow: 0 0 8px rgba(255, 71, 87, 0.7);
    animation: dot-blink 1.6s ease-in-out infinite;
}
.dot-pulse {
    background: var(--text-dim);
    box-shadow: 0 0 6px rgba(148, 160, 184, 0.5);
    animation: dot-pulse 2.4s ease-in-out infinite;
}
@keyframes dot-blink {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.25; }
}
@keyframes dot-pulse {
    0%, 100% { opacity: 0.5; transform: scale(0.85); }
    50%      { opacity: 1.0; transform: scale(1.15); }
}
.mono { font-family: var(--mono); }
.hint {
    color: var(--text-mute);
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    margin-left: auto;
    font-style: italic;
    animation: hint-pulse 2.5s ease-in-out infinite;
}
@keyframes hint-pulse {
    0%, 100% { opacity: 0.5; }
    50%      { opacity: 1.0; }
}

/* Viewfinder */
.viewfinder {
    position: relative;
    background: #000 !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: 6px !important;
    overflow: hidden;
    padding: 0 !important;
}
.viewfinder::before,
.viewfinder::after,
.viewfinder > div::before,
.viewfinder > div::after {
    content: "";
    position: absolute;
    width: 22px; height: 22px;
    border: 2px solid var(--alert);
    z-index: 5;
    pointer-events: none;
}
.viewfinder::before  { top: 8px;  left: 8px;
                       border-right: none; border-bottom: none; }
.viewfinder::after   { top: 8px;  right: 8px;
                       border-left:  none; border-bottom: none; }
.viewfinder > div::before { bottom: 8px; left: 8px;
                            border-right: none; border-top: none; }
.viewfinder > div::after  { bottom: 8px; right: 8px;
                            border-left:  none; border-top: none; }

.camera-feed, .camera-feed img, .camera-feed video {
    border-radius: 4px !important;
    width: 100% !important;
}

/* Telemetry strip */
.telemetry {
    margin-top: 0.65rem;
    padding: 0.55rem 0.85rem;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px;
    font-family: var(--mono);
    font-size: 0.78rem;
    display: flex;
    gap: 0.85rem;
    align-items: center;
    color: var(--text-dim);
    /* Keep size stable while values update. */
    height: 42px;
    box-sizing: border-box;
    white-space: nowrap;
    overflow: hidden;
}
.t-key  {
    color: var(--text-mute);
    flex-shrink: 0;
}
.t-val  {
    color: var(--text);
    font-weight: 600;
    flex-shrink: 0;
    min-width: 3.2em;
    display: inline-block;
}
.t-bar  {
    color: var(--accent, var(--text-dim));
    letter-spacing: -0.02em;
    transition: color 0.4s ease;
    flex-shrink: 0;
    min-width: 7em;
    display: inline-block;
}

/* Capture button */
.action-row { justify-content: center; margin-top: 0.85rem; }
.capture-btn button {
    background: transparent !important;
    color: var(--alert) !important;
    border: 1.5px solid var(--alert) !important;
    padding: 0.65rem 2.4rem !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.15em;
    font-family: var(--mono) !important;
    border-radius: 4px !important;
    cursor: pointer;
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
}
.capture-btn button::before {
    content: "";
    position: absolute;
    inset: 0;
    background: rgba(255, 71, 87, 0.08);
    transform: translateX(-100%);
    transition: transform 0.3s ease;
}
.capture-btn button:hover {
    background: rgba(255, 71, 87, 0.1) !important;
    box-shadow: 0 0 20px rgba(255, 71, 87, 0.35);
}
.capture-btn button:hover::before { transform: translateX(0); }

/* Status ring */
.status-shell {
    display: flex; justify-content: center; align-items: center;
    padding: 0.4rem;
    margin-bottom: 1rem;
    /* Keep widget height stable. */
    height: 175px;
    box-sizing: border-box;
}
.status-ring {
    position: relative;
    width: 150px; height: 150px;
    flex-shrink: 0;
}
.ring-svg {
    width: 100%; height: 100%;
    transform: rotate(-90deg);
}
.ring-track {
    fill: none;
    stroke: var(--border);
    stroke-width: 5;
}
.ring-fill {
    fill: none;
    stroke: var(--accent);
    stroke-width: 5;
    stroke-linecap: round;
    stroke-dasharray: 276;
    transition: stroke-dashoffset 0.5s ease, stroke 0.4s ease;
    filter: drop-shadow(0 0 6px var(--accent));
}
/* Slow rotating outer ring */
.status-ring::before {
    content: "";
    position: absolute;
    inset: -8px;
    border: 1px dashed var(--border-hi);
    border-radius: 50%;
    animation: ring-rotate 18s linear infinite;
    opacity: 0.4;
}
@keyframes ring-rotate {
    100% { transform: rotate(360deg); }
}
.status-text {
    position: absolute; inset: 0;
    display: flex; flex-direction: column;
    justify-content: center; align-items: center;
    gap: 0.15rem;
}
.status-label {
    font-family: var(--mono);
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: 0.1em;
    text-shadow: 0 0 8px var(--accent);
    transition: color 0.3s ease;
    /* Fixed width avoids layout shift across statuses. */
    min-width: 7.5em;
    text-align: center;
    font-variant-numeric: tabular-nums;
}
.status-conf {
    font-family: var(--mono);
    font-size: 0.75rem;
    color: var(--text-dim);
    letter-spacing: 0.05em;
    /* Fixed width avoids jumps when confidence changes. */
    min-width: 5em;
    text-align: center;
    font-variant-numeric: tabular-nums;
}

/* Message card */
.message-card {
    position: relative;
    background:
        linear-gradient(135deg, rgba(255,255,255,0.02), rgba(255,255,255,0)),
        var(--bg-card);
    border: 1px solid var(--border-hi);
    border-radius: 4px;
    /* Fixed height keeps the right panel stable. */
    height: 160px;
    min-height: 160px;
    max-height: 160px;
    box-sizing: border-box;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
    overflow: hidden;
}
.message-card::before {
    /* Edge glow follows --accent. */
    content: "";
    position: absolute; inset: 0;
    border: 1px solid var(--accent);
    opacity: 0.5;
    border-radius: 4px;
    pointer-events: none;
    box-shadow: inset 0 0 24px rgba(255, 255, 255, 0.02),
                0 0 16px -4px var(--accent);
}
.message-card .corner {
    position: absolute;
    width: 12px; height: 12px;
    border: 1.5px solid var(--accent);
}
.message-card .tl { top: 6px; left: 6px;
                    border-right: none; border-bottom: none; }
.message-card .tr { top: 6px; right: 6px;
                    border-left: none; border-bottom: none; }
.message-card .bl { bottom: 6px; left: 6px;
                    border-right: none; border-top: none; }
.message-card .br { bottom: 6px; right: 6px;
                    border-left: none; border-top: none; }
.message-text {
    /* Absolute positioning prevents reflow from long text. */
    position: absolute;
    top: 1.5rem;
    bottom: 2.5rem;
    left: 1.25rem;
    right: 1.25rem;
    /* Grid centering is robust with overflow constraints. */
    display: grid;
    place-items: center;
    text-align: center;
    color: var(--text);
    font-size: 1.25rem;
    font-weight: 700;
    line-height: 1.35;
    overflow: hidden;
    word-break: break-word;
    hyphens: auto;
    animation: text-fade-in 0.4s ease-out;
}
.message-meta {
    position: absolute;
    bottom: 0.65rem;
    right: 1rem;
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--accent);
    letter-spacing: 0.1em;
    opacity: 0.7;
}
@keyframes text-fade-in {
    0%   { opacity: 0; transform: translateY(2px); letter-spacing: 0.02em; }
    100% { opacity: 1; transform: translateY(0);  letter-spacing: 0;       }
}

/* Confidence bar */
.confidence-slider {
    background: var(--bg-card) !important;
    padding: 0.75rem 1rem !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
}
.confidence-slider input[type="range"] {
    accent-color: var(--accent) !important;
}

/* Accordions */
.top-accordion, .about-panel {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    margin-top: 0.85rem;
}
.top-accordion summary,
.about-panel summary {
    color: var(--text-dim) !important;
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em;
    padding: 0.6rem 0.85rem !important;
}
.top-accordion summary:hover,
.about-panel summary:hover {
    color: var(--text) !important;
}

/* Top-prediction label */
.gr-label, .label-wrap { background: transparent !important; }
.gr-label .progress { background: var(--accent) !important; }

/* Footer */
.hud-footer {
    margin-top: 2rem;
    padding-top: 0.85rem;
    border-top: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--text-mute);
    letter-spacing: 0.12em;
    text-align: center;
}
.footer-sep { padding: 0 0.5rem; color: var(--alert); }

/* Gradio overrides */
.gr-box, .gr-form, .gr-panel {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
.block.padded { background: transparent !important; }
label, .svelte-1gfkn6j { color: var(--text-dim) !important; }

/* Accessibility: visible focus rings */
button:focus-visible, [role="button"]:focus-visible {
    outline: 2px solid var(--alert) !important;
    outline-offset: 3px !important;
    border-radius: 4px;
}

/* Responsive */
@media (max-width: 900px) {
    .hud-header { flex-direction: column; align-items: flex-start; gap: 0.65rem; }
    .console-row { flex-direction: column !important; }
    .status-ring { width: 120px; height: 120px; }
    .message-text { font-size: 1.2rem; }
    .brand-name { font-size: 1.6rem; }
}
"""
