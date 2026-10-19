# Video2Site

Upload a screen-recording of a website; get a generated, responsive HTML/CSS/JS project, a sandboxed live preview, and a downloadable ZIP.

## Architecture
```
Browser dashboard (frontend/)  --XHR upload-->  Starlette API (backend/main.py)
                                                  |- validate: extension, magic bytes, size, duration (ffprobe), per-IP rate limit
                                                  |- extract frames: ffmpeg (OpenCV fallback)
                                                  |- analyze: Anthropic vision (if ANTHROPIC_API_KEY) else OpenCV palette heuristics
                                                  |- generate: Anthropic code model else deterministic template
                                                  |- package: project/ + ZIP; preview served with sandbox CSP
```
Generated code is never executed on the server; the preview runs in `<iframe sandbox="allow-scripts">` (no same-origin).
Java is intentionally not used (no requirement for it; one Python backend avoids duplicated services).

## Setup
```
pip install -r backend/requirements.txt      # also needs ffmpeg/ffprobe on PATH (recommended)
cp .env.example .env && export $(grep -v '^#' .env | sed 's/ *#.*//' | xargs)   # optional
cd backend && uvicorn main:app --port 8000   # open http://127.0.0.1:8000
```
Note: backend uses Starlette (the ASGI core of FastAPI), so `pip install fastapi` is not needed.

## Testing
1. `curl -F video=@file.txt localhost:8000/api/upload` -> 415 (bad extension)
2. Rename a text file to `.mp4` and upload -> 415 (magic-byte check)
3. Upload a real MP4/MOV/WebM, click **Analyze Video**; check Preview / AI analysis / Code tabs, device toggles, overlay, edit + Apply, Download Project.
4. Unzip the download and run `python backend/main.py`.

## Limits / honesty
- Without an API key the analysis is statistical (colors, brightness) and copy/layout are placeholders.
- Pixel-diff comparison is not implemented; a reference-frame overlay with opacity slider is provided.
- Per-section regeneration is not implemented; **Regenerate Design** regenerates the whole site using the edit prompt.
- If `API_TOKEN` is set, reference-frame images/video preview (plain `<img>` tags) can't send the header.
