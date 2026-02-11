from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.api.llava import router as llava_router

app = FastAPI(title="multimodal-anomaly-report-generation")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# /media 로 파일 서빙 (MEDIA_DIR 아래)
app.mount("/media", StaticFiles(directory=settings.media_dir), name="media")

app.include_router(llava_router)

@app.get("/health")
def health():
    return {"ok": True}
