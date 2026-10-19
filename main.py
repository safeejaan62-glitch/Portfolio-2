"""Video2Site API server (Starlette/ASGI). Run:  uvicorn main:app --port 8000   (from backend/)"""
import io
import json
import re
import shutil
import threading
import time
import traceback
import uuid
import zipfile
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

import config
import pipeline
from pipeline import PipelineError

JOB_ID = re.compile(r"^[0-9a-f]{32}$")
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()
POOL = ThreadPoolExecutor(max_workers=2)
HITS: dict[str, deque] = defaultdict(deque)
EDITABLE = {"index.html", "styles.css", "script.js"}


def err(msg: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": msg}, status_code=status)


# ------------------------------------------------------------------ middleware
class Guard(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/api/") and config.API_TOKEN:
            sent = request.headers.get("x-api-token", "")
            if not _consteq(sent, config.API_TOKEN):
                return err("Missing or invalid API token.", 401)
        resp = await call_next(request)
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        return resp


def _consteq(a: str, b: str) -> bool:
    import hmac
    return hmac.compare_digest(a.encode(), b.encode())


def rate_limited(ip: str) -> bool:
    now, q = time.time(), HITS[ip]
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= config.UPLOADS_PER_HOUR:
        return True
    q.append(now)
    return False


# ------------------------------------------------------------------ helpers
def get_job(request: Request):
    jid = request.path_params["job_id"]
    if not JOB_ID.match(jid) or jid not in JOBS:
        return None
    return JOBS[jid]


def set_job(job: dict, **kw):
    with LOCK:
        job.update(kw)
        job["updated"] = time.time()


def valid_magic(head: bytes, ext: str) -> bool:
    if ext == ".webm":
        return head[:4] == b"\x1a\x45\xdf\xa3"
    return head[4:8] in (b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip")


# ------------------------------------------------------------------ background pipeline
def run_job(job: dict, prompt: str, regenerate_only: bool = False):
    jdir = Path(job["dir"])
    try:
        if not regenerate_only:
            set_job(job, status="extracting", progress=10, message="Extracting key frames")
            meta = pipeline.probe(Path(job["video"]))
            if meta["duration"] > config.MAX_VIDEO_SECONDS:
                raise PipelineError(f"Video is {meta['duration']:.0f}s; the limit is {config.MAX_VIDEO_SECONDS:.0f}s.")
            n = 6 if meta["duration"] < 20 else config.MAX_FRAMES
            frames = pipeline.extract_frames(Path(job["video"]), jdir / "frames", meta["duration"], n)
            set_job(job, meta=meta, frames=[f.name for f in frames], progress=30)

            set_job(job, status="analyzing", message="Analyzing interface", progress=40)
            warnings = []
            if config.ANTHROPIC_API_KEY:
                try:
                    analysis = pipeline.llm_analysis(frames, meta)
                except Exception as e:  # noqa: BLE001
                    warnings.append(f"AI analysis failed ({e}); used offline analysis instead.")
                    analysis = pipeline.heuristic_analysis(frames, meta)
            else:
                analysis = pipeline.heuristic_analysis(frames, meta)
            (jdir / "analysis.json").write_text(json.dumps(analysis, indent=2))
            set_job(job, analysis=analysis, warnings=warnings, progress=60)
        else:
            frames = [jdir / "frames" / f for f in job["frames"]]
            analysis = job["analysis"]
            warnings = list(job.get("warnings", []))

        set_job(job, status="generating", message="Generating website code", progress=70)
        files = None
        if config.ANTHROPIC_API_KEY:
            try:
                files = pipeline.llm_codegen(analysis, prompt)
            except Exception as e:  # noqa: BLE001
                warnings.append(f"AI code generation failed ({e}); used template generator instead.")
        if files is None:
            files = pipeline.template_codegen(analysis, prompt)

        set_job(job, status="packaging", message="Packaging project", progress=90)
        proj = pipeline.build_project(jdir, files, analysis, frames)
        set_job(job, status="done", progress=100, message="Done", warnings=warnings,
                version=job.get("version", 0) + 1, files=pipeline.read_project_files(proj))
    except PipelineError as e:
        set_job(job, status="error", message=str(e))
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        set_job(job, status="error", message="Unexpected processing error. Please try another video.")


# ------------------------------------------------------------------ routes
async def upload(request: Request):
    ip = request.client.host if request.client else "?"
    if rate_limited(ip):
        return err("Too many uploads. Try again later.", 429)
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > config.MAX_UPLOAD_BYTES + 1_000_000:
        return err(f"File too large (max {config.MAX_UPLOAD_BYTES // 1024 // 1024} MB).", 413)
    try:
        form = await request.form()
    except Exception:  # noqa: BLE001
        return err("Malformed upload.")
    f = form.get("video")
    if f is None or not hasattr(f, "filename"):
        return err("No video file in the request (field name 'video').")
    ext = Path(f.filename or "").suffix.lower()
    if ext not in config.ALLOWED_EXT:
        return err("Unsupported format. Use MP4, MOV or WebM.", 415)
    jid = uuid.uuid4().hex
    jdir = config.JOBS_DIR / jid
    jdir.mkdir(parents=True)
    dest = jdir / f"upload{ext}"
    size = 0
    with open(dest, "wb") as out:
        while chunk := await f.read(1024 * 1024):
            size += len(chunk)
            if size > config.MAX_UPLOAD_BYTES:
                out.close()
                shutil.rmtree(jdir, ignore_errors=True)
                return err(f"File too large (max {config.MAX_UPLOAD_BYTES // 1024 // 1024} MB).", 413)
            out.write(chunk)
    with open(dest, "rb") as fh:
        head = fh.read(16)
    if size == 0 or not valid_magic(head, ext):
        shutil.rmtree(jdir, ignore_errors=True)
        return err("File content does not match a valid video of that type.", 415)
    job = {"id": jid, "dir": str(jdir), "video": str(dest), "status": "uploaded", "progress": 5,
           "message": "Upload complete", "created": time.time(), "updated": time.time(), "version": 0,
           "name": Path(f.filename).name[:120]}
    JOBS[jid] = job
    return JSONResponse({"job_id": jid, "mode": "ai" if config.ANTHROPIC_API_KEY else "offline"})


def public(job: dict) -> dict:
    keys = ["id", "status", "progress", "message", "meta", "frames", "analysis", "warnings", "version", "name"]
    return {k: job.get(k) for k in keys}


async def analyze(request: Request):
    job = get_job(request)
    if not job:
        return err("Unknown job.", 404)
    if job["status"] not in ("uploaded", "error", "done"):
        return err("Job already running.", 409)
    body = await _json(request)
    set_job(job, status="queued", progress=8, message="Queued")
    POOL.submit(run_job, job, str(body.get("prompt", ""))[:2000])
    return JSONResponse(public(job), status_code=202)


async def regenerate(request: Request):
    job = get_job(request)
    if not job or not job.get("analysis"):
        return err("Nothing to regenerate yet.", 404)
    if job["status"] not in ("done", "error"):
        return err("Job is busy.", 409)
    body = await _json(request)
    set_job(job, status="queued", progress=65, message="Queued")
    POOL.submit(run_job, job, str(body.get("prompt", ""))[:2000], True)
    return JSONResponse(public(job), status_code=202)


async def status(request: Request):
    job = get_job(request)
    return JSONResponse(public(job)) if job else err("Unknown job.", 404)


async def files(request: Request):
    job = get_job(request)
    if not job or not job.get("files"):
        return err("No generated files yet.", 404)
    return JSONResponse(job["files"])


async def save_file(request: Request):
    job = get_job(request)
    name = request.path_params["name"]
    if not job or job["status"] != "done":
        return err("Job not ready.", 409)
    if name not in EDITABLE:
        return err("Only index.html, styles.css and script.js can be edited.", 400)
    body = await _json(request)
    content = body.get("content")
    if not isinstance(content, str) or len(content) > 500_000:
        return err("Invalid content.")
    (Path(job["dir"]) / "project" / name).write_text(content, encoding="utf-8")
    job["files"][name] = content
    set_job(job, version=job["version"] + 1)
    return JSONResponse({"ok": True, "version": job["version"]})


async def preview(request: Request):
    """Self-contained HTML (CSS/JS inlined). Served with a sandbox CSP so it cannot touch the dashboard,
    cookies or storage and cannot make network requests."""
    job = get_job(request)
    if not job or not job.get("files"):
        return Response("Not generated yet.", status_code=404)
    f = job["files"]
    html = f["index.html"]
    html = re.sub(r'<link[^>]+href=["\']styles\.css["\'][^>]*>', lambda m: f"<style>{f['styles.css']}</style>", html)
    html = re.sub(r'<script[^>]+src=["\']script\.js["\'][^>]*>\s*</script>',
                  lambda m: "<script>" + f["script.js"].replace("</script", "<\\/script") + "</script>", html)
    csp = ("sandbox allow-scripts; default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
           "img-src data:; font-src data:; form-action 'none'; base-uri 'none'")
    return HTMLResponse(html, headers={"Content-Security-Policy": csp, "Cache-Control": "no-store"})


async def download(request: Request):
    job = get_job(request)
    if not job or job["status"] != "done":
        return err("Project not ready.", 404)
    proj = Path(job["dir"]) / "project"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(proj.rglob("*")):
            if p.is_file():
                z.write(p, Path("project") / p.relative_to(proj))
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="video2site-project.zip"'})


async def frame(request: Request):
    job = get_job(request)
    name = request.path_params["name"]
    if not job or not re.fullmatch(r"\d{2}\.jpg", name):
        return err("Not found.", 404)
    p = Path(job["dir"]) / "frames" / name
    return FileResponse(p) if p.exists() else err("Not found.", 404)


async def video(request: Request):
    job = get_job(request)
    if not job:
        return err("Not found.", 404)
    return FileResponse(job["video"])


async def contact(request: Request):
    """Demo endpoint so the dashboard's own preview/contact form has a target; stores nothing."""
    return JSONResponse({"ok": True})


async def config_info(request: Request):
    return JSONResponse({"mode": "ai" if config.ANTHROPIC_API_KEY else "offline",
                         "max_upload_mb": config.MAX_UPLOAD_BYTES // 1024 // 1024,
                         "max_seconds": config.MAX_VIDEO_SECONDS,
                         "auth_required": bool(config.API_TOKEN)})


async def _json(request: Request) -> dict:
    try:
        d = await request.json()
        return d if isinstance(d, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def cleanup_loop():
    while True:
        time.sleep(600)
        cutoff = time.time() - 6 * 3600
        for jid, j in list(JOBS.items()):
            if j["updated"] < cutoff and j["status"] in ("done", "error"):
                shutil.rmtree(j["dir"], ignore_errors=True)
                JOBS.pop(jid, None)


threading.Thread(target=cleanup_loop, daemon=True).start()

J = "/api/jobs/{job_id}"
app = Starlette(
    routes=[
        Route("/api/config", config_info),
        Route("/api/upload", upload, methods=["POST"]),
        Route(J + "/analyze", analyze, methods=["POST"]),
        Route(J + "/regenerate", regenerate, methods=["POST"]),
        Route(J, status),
        Route(J + "/files", files),
        Route(J + "/files/{name}", save_file, methods=["PUT"]),
        Route(J + "/preview", preview),
        Route(J + "/download", download),
        Route(J + "/frames/{name}", frame),
        Route(J + "/video", video),
        Route("/api/contact", contact, methods=["POST"]),
        Mount("/", StaticFiles(directory=config.FRONTEND_DIR, html=True)),
    ],
    middleware=[],
)
app.add_middleware(Guard)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
