#!/usr/bin/env python3
"""Build self-contained HTML from RESEARCH_BRIEF.md with embedded figures."""
import base64
import re
from pathlib import Path

import markdown

BRIEF_DIR = Path(__file__).resolve().parent
MD_FILE = BRIEF_DIR / "RESEARCH_BRIEF.md"
OUT_HTML = BRIEF_DIR / "research_brief.html"
FIG_DIR = BRIEF_DIR / "figures"

CSS = """
:root {
    --bg-primary: #0a0a0f;
    --bg-secondary: #1a1a2e;
    --bg-card: #16213e;
    --text-primary: #e0e0e0;
    --text-secondary: #a0a0a0;
    --accent-cyan: #06b6d4;
    --accent-magenta: #ff00ff;
    --border: #2a2a4a;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    background: var(--bg-primary);
    color: var(--text-primary);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 16px;
    line-height: 1.7;
    max-width: 900px;
    margin: 0 auto;
    padding: 3rem 2rem;
}

h1 {
    font-size: 2.2rem;
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-magenta));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
    line-height: 1.3;
}

h2 {
    color: var(--accent-cyan);
    font-size: 1.5rem;
    margin-top: 2.5rem;
    margin-bottom: 1rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid var(--border);
}

h3 {
    color: var(--accent-magenta);
    font-size: 1.2rem;
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
}

p { margin-bottom: 1rem; }

strong { color: #ffffff; }

a {
    color: var(--accent-cyan);
    text-decoration: none;
}
a:hover { text-decoration: underline; }

hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 2rem 0;
}

blockquote {
    border-left: 3px solid var(--accent-magenta);
    padding-left: 1rem;
    color: var(--text-secondary);
    margin: 1rem 0;
}

code {
    background: var(--bg-secondary);
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.9em;
    color: var(--accent-cyan);
}

pre {
    background: var(--bg-secondary);
    padding: 1rem;
    border-radius: 8px;
    overflow-x: auto;
    margin: 1rem 0;
    border: 1px solid var(--border);
}
pre code {
    padding: 0;
    background: none;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
    font-size: 0.9em;
}

th {
    background: var(--bg-card);
    color: var(--accent-cyan);
    padding: 0.6rem 0.8rem;
    text-align: left;
    border-bottom: 2px solid var(--accent-cyan);
}

td {
    padding: 0.5rem 0.8rem;
    border-bottom: 1px solid var(--border);
}

tr:hover { background: var(--bg-secondary); }

img {
    max-width: 100%;
    border-radius: 8px;
    margin: 1rem 0;
    border: 1px solid var(--border);
}

ol, ul {
    margin: 0.5rem 0 1rem 1.5rem;
}

li { margin-bottom: 0.4rem; }

.metadata {
    color: var(--text-secondary);
    font-size: 0.9em;
    margin-bottom: 2rem;
}
"""


def embed_images(html: str) -> str:
    """Replace img src with base64 data URIs."""
    def replace_img(match: re.Match) -> str:
        src = match.group(1)
        # Resolve relative to figures dir
        img_path = FIG_DIR / Path(src).name
        if not img_path.exists():
            # Try relative to brief dir
            img_path = BRIEF_DIR / src
        if not img_path.exists():
            return match.group(0)  # leave as-is

        suffix = img_path.suffix.lower()
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".pdf": "application/pdf", ".svg": "image/svg+xml"}.get(suffix, "image/png")

        data = base64.b64encode(img_path.read_bytes()).decode()
        return f'src="data:{mime};base64,{data}"'

    return re.sub(r'src="([^"]+)"', replace_img, html)


def main():
    md_text = MD_FILE.read_text()

    # Convert markdown to HTML
    extensions = ["tables", "fenced_code", "toc", "smarty"]
    html_body = markdown.markdown(md_text, extensions=extensions)

    # Embed images
    html_body = embed_images(html_body)

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LENS V2: Memory Strategy Ablation — Research Brief</title>
    <style>{CSS}</style>
</head>
<body>
{html_body}
</body>
</html>"""

    OUT_HTML.write_text(html_doc)
    size_kb = OUT_HTML.stat().st_size / 1024
    print(f"Written: {OUT_HTML} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
