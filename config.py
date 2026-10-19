import os
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE / "data"))
JOBS_DIR = DATA_DIR / "jobs"
FRONTEND_DIR = BASE.parent / "frontend"

MAX_UPLOAD_BYTES = int(float(os.getenv("MAX_UPLOAD_MB", "200")) * 1024 * 1024)
MAX_VIDEO_SECONDS = float(os.getenv("MAX_VIDEO_SECONDS", "180"))
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "8"))
UPLOADS_PER_HOUR = int(os.getenv("UPLOADS_PER_HOUR", "20"))
FFMPEG_TIMEOUT = 90
API_TOKEN = os.getenv("API_TOKEN", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
ALLOWED_EXT = {".mp4", ".mov", ".webm"}
JOBS_DIR.mkdir(parents=True, exist_ok=True)
