import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Dict

import trimesh


class TripoSRService:
    """Wrapper chạy TripoSR để tái tạo object 360 từ một ảnh đơn."""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.triposr_dir = Path(os.getenv("TRIPOSR_DIR", "")).expanduser()
        self.mc_resolution = int(os.getenv("TRIPOSR_MC_RESOLUTION", "256"))
        self.chunk_size = int(os.getenv("TRIPOSR_CHUNK_SIZE", "4096"))
        self.foreground_ratio = float(os.getenv("TRIPOSR_FOREGROUND_RATIO", "0.85"))

    @property
    def available(self) -> bool:
        return bool(str(self.triposr_dir)) and (self.triposr_dir / "run.py").exists()

    def ensure_available(self) -> None:
        if not self.available:
            raise RuntimeError(
                "Chưa cấu hình TripoSR. Hãy clone https://github.com/VAST-AI-Research/TripoSR "
                "và đặt biến môi trường TRIPOSR_DIR trỏ tới thư mục đó."
            )

    def run(self, image_path: str) -> Dict[str, str]:
        """Chạy TripoSR run.py và gom GLB/OBJ/PLY vào thư mục outputs của web app."""
        self.ensure_available()

        stem = Path(image_path).stem or "image"
        job_dir = self.output_dir / f"{stem}_triposr_{uuid.uuid4().hex[:8]}"
        raw_dir = job_dir / "_triposr_raw"
        job_dir.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable,
            "run.py",
            str(Path(image_path).resolve()),
            "--output-dir",
            str(raw_dir.resolve()),
            "--model-save-format",
            "glb",
            "--mc-resolution",
            str(self.mc_resolution),
            "--chunk-size",
            str(self.chunk_size),
            "--foreground-ratio",
            str(self.foreground_ratio),
        ]

        result = subprocess.run(
            command,
            cwd=str(self.triposr_dir),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"TripoSR lỗi khi chạy inference: {detail[-2000:]}")

        raw_glb = raw_dir / "0" / "mesh.glb"
        if not raw_glb.exists():
            raise RuntimeError("TripoSR chạy xong nhưng không tìm thấy file output mesh.glb.")

        glb_path = job_dir / "mesh.glb"
        obj_path = job_dir / "mesh.obj"
        ply_path = job_dir / "mesh.ply"
        shutil.copy2(raw_glb, glb_path)

        mesh = trimesh.load(str(glb_path), force="mesh")
        mesh.export(str(obj_path))
        mesh.export(str(ply_path))

        return {
            "job_id": job_dir.name,
            "glb": str(glb_path),
            "obj": str(obj_path),
            "ply": str(ply_path),
        }
