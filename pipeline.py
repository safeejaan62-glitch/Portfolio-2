"""Video -> frames -> structured analysis -> website code.

Two modes:
  * LLM mode  (ANTHROPIC_API_KEY set): vision model describes the UI, coding model writes the site.
  * Offline   (no key): OpenCV colour/brightness analysis + deterministic template generator.
Generated code is never executed here; it is only stored, zipped and served into a sandboxed iframe.
"""
import base64
import json
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np
import requests

import config


class PipelineError(Exception):
    pass


# ---------------------------------------------------------------- video probing / frames
def probe(video: Path) -> dict:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,codec_name:format=duration",
             "-of", "json", str(video)],
            capture_output=True, text=True, timeout=30)
        info = json.loads(r.stdout or "{}")
        stream = (info.get("streams") or [None])[0]
        if not stream:
            raise PipelineError("No video stream found. The file may be corrupt or not a real video.")
        return {"width": stream["width"], "height": stream["height"],
                "codec": stream["codec_name"],
                "duration": float(info["format"]["duration"])}
    except PipelineError:
        raise
    except Exception as e:  # noqa: BLE001
        raise PipelineError(f"Could not read video metadata: {e}")


def extract_frames(video: Path, out_dir: Path, duration: float, n: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = max(1, min(n, config.MAX_FRAMES))
    times = [duration * (i + 0.5) / n for i in range(n)]
    frames = []
    for i, t in enumerate(times):
        dest = out_dir / f"{i + 1:02d}.jpg"
        try:
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", str(video),
                 "-frames:v", "1", "-vf", "scale='min(960,iw)':-2", "-q:v", "4", str(dest)],
                check=True, timeout=config.FFMPEG_TIMEOUT, capture_output=True)
        except Exception:  # fall back to OpenCV for this frame
            cap = cv2.VideoCapture(str(video))
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, img = cap.read()
            cap.release()
            if ok:
                h, w = img.shape[:2]
                if w > 960:
                    img = cv2.resize(img, (960, int(h * 960 / w)))
                cv2.imwrite(str(dest), img)
        if dest.exists() and dest.stat().st_size > 0:
            frames.append(dest)
    if not frames:
        raise PipelineError("Could not extract any frames from the video.")
    return frames


# ---------------------------------------------------------------- offline analysis
def _hex(bgr) -> str:
    b, g, r = [int(x) for x in bgr]
    return f"#{r:02x}{g:02x}{b:02x}"


def heuristic_analysis(frames: list[Path], meta: dict) -> dict:
    samples, brightness = [], []
    for f in frames:
        img = cv2.imread(str(f))
        if img is None:
            continue
        small = cv2.resize(img, (64, 64))
        samples.append(small.reshape(-1, 3))
        brightness.append(float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean()))
    if not samples:
        raise PipelineError("Frames could not be decoded.")
    data = np.vstack(samples).astype(np.float32)
    k = 6
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    hsv = cv2.cvtColor(centers.reshape(1, k, 3).astype(np.uint8), cv2.COLOR_BGR2HSV).reshape(k, 3)
    order = np.argsort(-counts)
    # vivid colours (saturation*value) become primary/secondary/accent
    vivid = sorted(range(k), key=lambda i: -(hsv[i][1] / 255) * (hsv[i][2] / 255) * (1 + counts[i] / counts.sum()))
    dark_idx = min(range(k), key=lambda i: hsv[i][2])
    avg_b = float(np.mean(brightness))
    mode = "dark" if avg_b < 110 else "light"
    palette = {
        "primary": _hex(centers[vivid[0]]),
        "secondary": _hex(centers[vivid[1]]),
        "accent": _hex(centers[vivid[2]]),
        "background": "#0b0b12" if mode == "dark" else "#faf7f4",
        "surface": "#161622" if mode == "dark" else "#ffffff",
        "text": "#f4f1ee" if mode == "dark" else "#1d1b1a",
    }
    return {
        "title": "Creative Portfolio",
        "summary": ("Immersive portfolio-style site with a full-screen hero, project showcase, "
                    "about, services, FAQ and contact. Inferred from colour and layout statistics only "
                    "(offline mode - set ANTHROPIC_API_KEY for true visual analysis)."),
        "mode": mode,
        "palette": palette,
        "dominant_colors": [_hex(centers[i]) for i in order],
        "darkest_color": _hex(centers[dark_idx]),
        "average_brightness": round(avg_b, 1),
        "fonts": {"heading": "Georgia, 'Times New Roman', serif",
                  "body": "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"},
        "navigation": ["Work", "About", "Services", "FAQ", "Contact"],
        "sections": [
            {"id": "work", "type": "gallery", "heading": "Selected Work",
             "text": "A showcase of recent projects.",
             "items": [{"title": f"Project {i}", "text": "Case study summary."} for i in range(1, 7)]},
            {"id": "about", "type": "about", "heading": "About",
             "text": "Designer and developer building memorable, interactive web experiences."},
            {"id": "services", "type": "features", "heading": "Services",
             "items": [{"title": "Web Design", "text": "Interfaces with a clear visual identity."},
                       {"title": "Development", "text": "Fast, accessible, responsive builds."},
                       {"title": "Project Management", "text": "Clear scope, steady delivery."}]},
            {"id": "faq", "type": "faq", "heading": "FAQ",
             "items": [{"title": "How do we start?", "text": "Send a message with your goals."},
                       {"title": "How long does a project take?", "text": "Usually 3 to 8 weeks."}]},
            {"id": "contact", "type": "contact", "heading": "Get in touch", "text": "Tell me about your project."},
        ],
        "interactions": ["Sticky navigation", "Hover lift on cards", "Accordion FAQ", "Modal project viewer",
                         "Contact form validation", "Scroll reveal"],
        "animations": ["Fade-up on scroll", "Gradient drift in hero"],
        "responsive_notes": "Single column on mobile, 2-3 column grids on tablet/desktop, hamburger menu under 760px.",
        "assumptions": [
            "Offline mode cannot read on-screen text, icons or fonts; copy and typography are placeholders.",
            "Section structure is a standard portfolio layout, not detected from the video.",
            f"Colours come from k-means over {len(frames)} frames.",
            f"Video is {meta['width']}x{meta['height']} (portrait phone capture of a screen), so true page "
            "geometry and spacing could not be measured.",
        ],
        "mode_used": "offline",
    }


# ---------------------------------------------------------------- Anthropic API helpers
def _call_llm(system: str, content: list, max_tokens: int = 8000) -> str:
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": config.ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": config.ANTHROPIC_MODEL, "max_tokens": max_tokens, "system": system,
              "messages": [{"role": "user", "content": content}]},
        timeout=180)
    if r.status_code != 200:
        raise PipelineError(f"AI service error {r.status_code}: {r.text[:200]}")
    return "".join(b.get("text", "") for b in r.json().get("content", []))


ANALYSIS_SYSTEM = """You are a senior UI/UX analyst. You receive frames sampled from a screen recording of a website.
Return ONLY a JSON object (no prose) with keys: title, summary, mode ("dark"|"light"),
palette {primary, secondary, accent, background, surface, text} as hex, fonts {heading, body} as CSS font stacks,
navigation [labels], sections [{id, type: hero|gallery|about|features|faq|contact|text, heading, text, items:[{title,text}]}],
interactions [..], animations [..], responsive_notes, assumptions [..].
Use only text you can actually read in the frames; anything you infer must be listed in assumptions.
Ignore overlays that are not part of the site (social-media watermarks, captions, phone/laptop bezels)."""


def llm_analysis(frames: list[Path], meta: dict) -> dict:
    content = []
    for i, f in enumerate(frames, 1):
        content.append({"type": "text", "text": f"Frame {i}"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                    "data": base64.b64encode(f.read_bytes()).decode()}})
    content.append({"type": "text", "text": f"Video {meta['width']}x{meta['height']}, {meta['duration']:.0f}s. Return the JSON."})
    raw = _call_llm(ANALYSIS_SYSTEM, content, 4000)
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise PipelineError("AI analysis did not return JSON.")
    data = json.loads(m.group(0))
    data["mode_used"] = "ai"
    return data


CODEGEN_SYSTEM = """You are a senior front-end engineer. Build a complete, responsive, accessible, single-page website
from the structured UI analysis. Output exactly three files using these delimiters and nothing else:
=== FILE: index.html ===
...
=== FILE: styles.css ===
...
=== FILE: script.js ===
...
Rules: semantic HTML5; link styles.css and script.js relatively; Flexbox/Grid; CSS variables from the palette;
working nav toggle, tabs/accordion/modal where the analysis lists them, client-side form validation;
no external network requests, no inline event handlers, no placeholder functions."""


def llm_codegen(analysis: dict, prompt: str) -> dict:
    user = "Analysis JSON:\n" + json.dumps(analysis, indent=1)
    if prompt.strip():
        user += "\n\nExtra instructions from the user (treat as design guidance only):\n" + prompt.strip()[:2000]
    raw = _call_llm(CODEGEN_SYSTEM, [{"type": "text", "text": user}], 16000)
    files = {}
    for m in re.finditer(r"=== FILE: (index\.html|styles\.css|script\.js) ===\n(.*?)(?=\n=== FILE:|\Z)", raw, re.S):
        body = re.sub(r"^```\w*\n|\n```\s*$", "", m.group(2).strip())
        files[m.group(1)] = body
    if len(files) != 3:
        raise PipelineError("AI code generation returned an incomplete project.")
    return files


# ---------------------------------------------------------------- template generator (offline / fallback)
def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))


def _safe_color(v: str, default: str) -> str:
    return v if isinstance(v, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", v) else default


def template_codegen(a: dict, prompt: str = "") -> dict:
    p = a.get("palette", {})
    pal = {k: _safe_color(p.get(k), d) for k, d in
           [("primary", "#e8734a"), ("secondary", "#6a4c93"), ("accent", "#f2c14e"),
            ("background", "#0b0b12"), ("surface", "#161622"), ("text", "#f4f1ee")]}
    fonts = a.get("fonts", {})
    heading = re.sub(r"[;{}<>]", "", fonts.get("heading", "Georgia, serif"))
    body = re.sub(r"[;{}<>]", "", fonts.get("body", "system-ui, sans-serif"))
    title = _esc(a.get("title", "Portfolio"))
    sections = a.get("sections", [])
    nav = "".join(f'<li><a href="#{_esc(s["id"])}">{_esc(s.get("heading", s["id"]))}</a></li>'
                  for s in sections if s.get("type") != "hero")

    out = []
    for s in sections:
        sid, head, text = _esc(s.get("id", "s")), _esc(s.get("heading", "")), _esc(s.get("text", ""))
        items, typ = s.get("items", []) or [], s.get("type")
        if typ == "gallery":
            cards = "".join(
                f'<button class="card project" data-title="{_esc(i["title"])}" data-text="{_esc(i["text"])}" '
                f'style="--hue:{n * 47 % 360}"><span class="thumb"></span><h3>{_esc(i["title"])}</h3>'
                f'<p>{_esc(i["text"])}</p></button>' for n, i in enumerate(items, 1))
            out.append(f'<section id="{sid}" class="reveal"><h2>{head}</h2><p class="lead">{text}</p>'
                       f'<div class="grid">{cards}</div></section>')
        elif typ == "features":
            tabs = "".join(f'<button role="tab" aria-selected="{str(n == 0).lower()}" data-tab="{n}">{_esc(i["title"])}</button>'
                           for n, i in enumerate(items))
            panels = "".join(f'<div role="tabpanel" data-panel="{n}" {"" if n == 0 else "hidden"}><p>{_esc(i["text"])}</p></div>'
                             for n, i in enumerate(items))
            out.append(f'<section id="{sid}" class="reveal"><h2>{head}</h2>'
                       f'<div class="tabs"><div class="tablist" role="tablist">{tabs}</div>{panels}</div></section>')
        elif typ == "faq":
            qa = "".join(f'<div class="acc"><button aria-expanded="false">{_esc(i["title"])}</button>'
                         f'<div class="acc-body" hidden><p>{_esc(i["text"])}</p></div></div>' for i in items)
            out.append(f'<section id="{sid}" class="reveal"><h2>{head}</h2>{qa}</section>')
        elif typ == "contact":
            out.append(f'<section id="{sid}" class="reveal"><h2>{head}</h2><p class="lead">{text}</p>'
                       '<form id="contact" novalidate><label>Name<input name="name" required minlength="2"></label>'
                       '<label>Email<input name="email" type="email" required></label>'
                       '<label>Message<textarea name="message" rows="4" required minlength="10"></textarea></label>'
                       '<button class="btn" type="submit">Send message</button><p class="status" role="status"></p></form></section>')
        else:
            out.append(f'<section id="{sid}" class="reveal"><h2>{head}</h2><p class="lead">{text}</p></section>')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="styles.css">
</head>
<body>
<header class="site-header">
  <a class="logo" href="#top">{title}</a>
  <button class="menu-toggle" aria-label="Toggle menu" aria-expanded="false">&#9776;</button>
  <nav><ul id="nav-list">{nav}</ul></nav>
</header>
<main id="top">
  <section class="hero">
    <p class="eyebrow">Portfolio</p>
    <h1>{_esc(a.get("title", "Portfolio"))}</h1>
    <p class="lead">{_esc(re.split(r"(?<=[.!?])\s", a.get("summary", ""))[0][:180])}</p>
    <a class="btn" href="#{_esc(sections[0]["id"]) if sections else "top"}">View work</a>
  </section>
  {"".join(out)}
</main>
<div class="modal" hidden role="dialog" aria-modal="true" aria-labelledby="m-title">
  <div class="modal-box"><button class="modal-close" aria-label="Close">&times;</button>
  <h3 id="m-title"></h3><p id="m-text"></p></div>
</div>
<footer><p>&copy; <span id="year"></span> {title}</p></footer>
<script src="script.js"></script>
</body>
</html>
"""
    css = f""":root {{
  --primary:{pal['primary']}; --secondary:{pal['secondary']}; --accent:{pal['accent']};
  --bg:{pal['background']}; --surface:{pal['surface']}; --text:{pal['text']};
  --font-h:{heading}; --font-b:{body};
}}
*{{box-sizing:border-box}} html{{scroll-behavior:smooth}}
body{{margin:0;background:var(--bg);color:var(--text);font-family:var(--font-b);line-height:1.6}}
h1,h2,h3{{font-family:var(--font-h);line-height:1.15}}
a{{color:inherit}}
.site-header{{position:sticky;top:0;z-index:10;display:flex;align-items:center;justify-content:space-between;
  padding:14px 5vw;background:color-mix(in srgb,var(--bg) 85%,transparent);backdrop-filter:blur(10px);
  border-bottom:1px solid color-mix(in srgb,var(--text) 12%,transparent)}}
.logo{{font-family:var(--font-h);font-weight:700;text-decoration:none;font-size:1.2rem}}
nav ul{{display:flex;gap:28px;list-style:none;margin:0;padding:0}}
nav a{{text-decoration:none;opacity:.8;transition:opacity .2s,color .2s}} nav a:hover{{opacity:1;color:var(--accent)}}
.menu-toggle{{display:none;background:none;border:0;color:inherit;font-size:1.6rem;cursor:pointer}}
.hero{{min-height:78vh;display:flex;flex-direction:column;justify-content:center;align-items:flex-start;gap:16px;
  padding:8vh 5vw;background:linear-gradient(120deg,var(--secondary),var(--primary),var(--accent));
  background-size:200% 200%;animation:drift 14s ease infinite;color:#fff}}
.hero h1{{font-size:clamp(2.4rem,7vw,5rem);margin:0}} .hero .lead{{max-width:46ch}}
.eyebrow{{letter-spacing:.2em;text-transform:uppercase;font-size:.8rem;margin:0;opacity:.85}}
@keyframes drift{{0%{{background-position:0 50%}}50%{{background-position:100% 50%}}100%{{background-position:0 50%}}}}
section:not(.hero){{padding:80px 5vw;max-width:1180px;margin:auto}}
h2{{font-size:clamp(1.8rem,4vw,2.8rem);margin:0 0 8px}} .lead{{opacity:.8;max-width:60ch}}
.btn{{display:inline-block;padding:12px 26px;border:0;border-radius:999px;background:var(--text);color:var(--bg);
  font-weight:600;text-decoration:none;cursor:pointer;transition:transform .2s,box-shadow .2s}}
.btn:hover{{transform:translateY(-2px);box-shadow:0 10px 24px rgba(0,0,0,.35)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:22px;margin-top:32px}}
.card{{text-align:left;background:var(--surface);color:inherit;border:1px solid color-mix(in srgb,var(--text) 10%,transparent);
  border-radius:16px;padding:14px;cursor:pointer;font:inherit;transition:transform .25s,box-shadow .25s}}
.card:hover,.card:focus-visible{{transform:translateY(-6px);box-shadow:0 16px 34px rgba(0,0,0,.4)}}
.thumb{{display:block;aspect-ratio:16/10;border-radius:10px;margin-bottom:12px;
  background:linear-gradient(135deg,hsl(var(--hue) 70% 55%),hsl(calc(var(--hue) + 50) 70% 35%))}}
.card h3{{margin:0 0 4px}} .card p{{margin:0;opacity:.7}}
.tabs{{margin-top:24px}} .tablist{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}}
.tablist button{{padding:10px 18px;border-radius:999px;border:1px solid color-mix(in srgb,var(--text) 25%,transparent);
  background:none;color:inherit;cursor:pointer;font:inherit}}
.tablist button[aria-selected=true]{{background:var(--primary);border-color:var(--primary);color:#fff}}
[role=tabpanel]{{background:var(--surface);padding:22px;border-radius:14px}}
.acc{{border-bottom:1px solid color-mix(in srgb,var(--text) 15%,transparent)}}
.acc>button{{width:100%;text-align:left;background:none;border:0;color:inherit;font:inherit;font-weight:600;padding:18px 0;cursor:pointer}}
.acc>button::after{{content:"+";float:right;transition:transform .2s}} .acc>button[aria-expanded=true]::after{{transform:rotate(45deg)}}
.acc-body{{padding-bottom:16px;opacity:.8}}
form{{display:grid;gap:16px;max-width:520px;margin-top:24px}} label{{display:grid;gap:6px;font-size:.9rem}}
input,textarea{{padding:12px;border-radius:10px;border:1px solid color-mix(in srgb,var(--text) 25%,transparent);
  background:var(--surface);color:var(--text);font:inherit}}
input.invalid,textarea.invalid{{border-color:#ff6b6b}} .status{{min-height:1.4em;margin:0}}
.modal{{position:fixed;inset:0;background:rgba(0,0,0,.65);display:grid;place-items:center;z-index:50;padding:20px}}
.modal[hidden]{{display:none}} .modal-box{{position:relative;background:var(--surface);padding:32px;border-radius:18px;max-width:460px;width:100%}}
.modal-close{{position:absolute;top:10px;right:14px;background:none;border:0;color:inherit;font-size:1.8rem;cursor:pointer}}
footer{{text-align:center;padding:40px;opacity:.6}}
.reveal{{opacity:0;transform:translateY(24px);transition:opacity .7s,transform .7s}} .reveal.in{{opacity:1;transform:none}}
@media (max-width:760px){{
  .menu-toggle{{display:block}}
  nav{{position:absolute;top:100%;left:0;right:0;background:var(--bg);display:none}}
  nav.open{{display:block}} nav ul{{flex-direction:column;gap:0;padding:8px 5vw}} nav li a{{display:block;padding:12px 0}}
  section:not(.hero){{padding:56px 5vw}}
}}
@media (prefers-reduced-motion:reduce){{*{{animation:none!important;transition:none!important}}.reveal{{opacity:1;transform:none}}}}
"""
    js = """(() => {
  'use strict';
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  $('#year').textContent = new Date().getFullYear();

  const toggle = $('.menu-toggle'), nav = $('nav');
  toggle.addEventListener('click', () => {
    const open = nav.classList.toggle('open');
    toggle.setAttribute('aria-expanded', String(open));
  });
  $$('nav a').forEach(a => a.addEventListener('click', () => nav.classList.remove('open')));

  const io = 'IntersectionObserver' in window
    ? new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } }), { threshold: .12 })
    : null;
  $$('.reveal').forEach(el => io ? io.observe(el) : el.classList.add('in'));

  $$('.tabs').forEach(tabs => {
    const btns = $$('[role=tab]', tabs), panels = $$('[role=tabpanel]', tabs);
    btns.forEach((b, i) => b.addEventListener('click', () => {
      btns.forEach((x, j) => x.setAttribute('aria-selected', String(i === j)));
      panels.forEach((p, j) => { p.hidden = i !== j; });
    }));
  });

  $$('.acc > button').forEach(b => b.addEventListener('click', () => {
    const open = b.getAttribute('aria-expanded') === 'true';
    b.setAttribute('aria-expanded', String(!open));
    b.nextElementSibling.hidden = open;
  }));

  const modal = $('.modal');
  const closeModal = () => { modal.hidden = true; };
  $$('.project').forEach(c => c.addEventListener('click', () => {
    $('#m-title').textContent = c.dataset.title; $('#m-text').textContent = c.dataset.text; modal.hidden = false;
    $('.modal-close').focus();
  }));
  $('.modal-close').addEventListener('click', closeModal);
  modal.addEventListener('click', e => { if (e.target === modal) closeModal(); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

  const form = $('#contact');
  if (form) form.addEventListener('submit', async e => {
    e.preventDefault();
    const status = $('.status', form); let ok = true;
    $$('input,textarea', form).forEach(f => { const v = f.checkValidity(); f.classList.toggle('invalid', !v); ok = ok && v; });
    if (!ok) { status.textContent = 'Please fix the highlighted fields.'; return; }
    status.textContent = 'Sending...';
    try {
      const r = await fetch('/api/contact', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(Object.fromEntries(new FormData(form))) });
      if (!r.ok) throw new Error();
      status.textContent = 'Thanks! Your message was sent.'; form.reset();
    } catch { status.textContent = 'Could not reach the server (start backend/main.py).'; }
  });
})();
"""
    return {"index.html": html, "styles.css": css, "script.js": js}


# ---------------------------------------------------------------- project packaging
PROJECT_MAIN = '''"""Tiny dependency-free server for the generated site. Run: python backend/main.py"""
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def do_POST(self):
        if self.path != "/api/contact":
            return self.send_error(404)
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > 10_000:
                raise ValueError("too large")
            data = json.loads(self.rfile.read(n))
            if not all(isinstance(data.get(k), str) and data[k].strip() for k in ("name", "email", "message")):
                raise ValueError("missing fields")
            if "@" not in data["email"]:
                raise ValueError("bad email")
        except Exception:
            return self.send_error(400, "Invalid submission")
        with open(ROOT / "backend" / "messages.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({k: data[k][:2000] for k in ("name", "email", "message")}) + "\\n")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok": true}')


if __name__ == "__main__":
    print("Serving on http://127.0.0.1:8080")
    ThreadingHTTPServer(("127.0.0.1", 8080), Handler).serve_forever()
'''

PROJECT_README = """# {title}

Generated by Video2Site from a reference video.

## Run
```
python backend/main.py     # then open http://127.0.0.1:8080
```
Or open `index.html` directly (the contact form needs the backend).

## Files
- `index.html`, `styles.css`, `script.js` - the site (no external dependencies)
- `backend/main.py` - stdlib-only server + `/api/contact` endpoint (stores to `backend/messages.jsonl`)
- `backend/requirements.txt` - no third-party packages required
- `assets/` - reference frames extracted from your video

## Assumptions made
{assumptions}
"""


def build_project(job_dir: Path, files: dict, analysis: dict, frames: list[Path]) -> Path:
    proj = job_dir / "project"
    (proj / "backend").mkdir(parents=True, exist_ok=True)
    (proj / "assets").mkdir(exist_ok=True)
    for name, content in files.items():
        (proj / name).write_text(content, encoding="utf-8")
    (proj / "backend" / "main.py").write_text(PROJECT_MAIN, encoding="utf-8")
    (proj / "backend" / "requirements.txt").write_text("# No third-party dependencies (Python 3.9+ standard library only)\n")
    assumptions = "\n".join(f"- {x}" for x in analysis.get("assumptions", [])) or "- None recorded"
    (proj / "README.md").write_text(PROJECT_README.format(title=analysis.get("title", "Site"), assumptions=assumptions),
                                    encoding="utf-8")
    for f in frames:
        (proj / "assets" / f"reference-{f.name}").write_bytes(f.read_bytes())
    return proj


def read_project_files(proj: Path) -> dict:
    names = ["index.html", "styles.css", "script.js", "backend/main.py"]
    return {n: (proj / n).read_text(encoding="utf-8") for n in names if (proj / n).exists()}
