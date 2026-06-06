"""
Custom CSS for Silent-SOS Gradio app.
This is what prevents the app from looking like a "basic Gradio demo".
"""

CUSTOM_CSS = """
/* Emergency-app aesthetic: dark background, high-contrast alerts */
.gradio-container {
    background: #0a0e1a !important;
    color: #e8eaed !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    max-width: 1200px !important;
}

.header h1 {
    color: #ff4757 !important;
    font-size: 3rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
    margin-bottom: 0 !important;
}

.header h3 {
    color: #95a5b6 !important;
    font-weight: 400 !important;
}

.camera-feed {
    border: 2px solid #2c3e50 !important;
    border-radius: 12px !important;
    overflow: hidden;
}

.status-box textarea,
.message-box textarea {
    background: #1a1f2e !important;
    color: #e8eaed !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    text-align: center;
    border: 2px solid #2c3e50 !important;
    border-radius: 8px !important;
    padding: 1.5rem !important;
}

.status-box textarea[value="ABSTAINED"] { color: #f39c12 !important; }
.status-box textarea[value="OK"]        { color: #2ecc71 !important; }
.status-box textarea[value="ERROR"]     { color: #e74c3c !important; }

.footer {
    text-align: center;
    color: #6c7a89 !important;
    font-size: 0.85rem !important;
    margin-top: 2rem;
}

/* Accessibility: large, focusable controls */
button, .gr-button {
    min-height: 56px !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
}
"""
