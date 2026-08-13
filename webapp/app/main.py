import os
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from app.pipeline import ImageTo3DService
from app.triposr_pipeline import TripoSRService


BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = os.getenv("DEPTH_MODEL_ID", "depth-anything/Depth-Anything-V2-Small-hf")
service = ImageTo3DService(model_id=MODEL_ID, output_dir=str(OUTPUT_DIR))
triposr_service = TripoSRService(output_dir=str(OUTPUT_DIR))

app = FastAPI(title="Image2To3D API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": service.device,
        "depth_model": service.model_id,
        "triposr_available": triposr_service.available,
    }


@app.post("/api/generate")
async def generate_3d(file: UploadFile = File(...), mode: str = Form("depth")):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        raise HTTPException(status_code=400, detail="Chi ho tro file JPG/PNG")

    file_id = uuid.uuid4().hex[:8]
    local_path = UPLOAD_DIR / f"input_{file_id}{ext}"

    with local_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        if mode == "triposr":
            outputs = triposr_service.run(str(local_path))
            job_id = outputs["job_id"]
            return {
                "message": "Tao mo hinh 360 thanh cong",
                "job_id": job_id,
                "device": service.device,
                "model": "TripoSR",
                "mode": "triposr",
                "preview_glb_url": f"/outputs/{job_id}/mesh.glb",
                "download": {
                    "ply": f"/outputs/{job_id}/mesh.ply",
                    "obj": f"/outputs/{job_id}/mesh.obj",
                    "glb": f"/outputs/{job_id}/mesh.glb",
                },
            }

        outputs = service.run(str(local_path))
        job_id = outputs["job_id"]
        return {
            "message": "Tao mo hinh 3D thanh cong",
            "job_id": job_id,
            "device": service.device,
            "model": service.model_id,
            "mode": "depth",
            "preview_glb_url": f"/outputs/{job_id}/mesh.glb",
            "download": {
                "depth_png": f"/outputs/{job_id}/depth.png",
                "ply": f"/outputs/{job_id}/point_cloud.ply",
                "obj": f"/outputs/{job_id}/mesh.obj",
                "glb": f"/outputs/{job_id}/mesh.glb",
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Loi xu ly 3D: {str(exc)}") from exc
