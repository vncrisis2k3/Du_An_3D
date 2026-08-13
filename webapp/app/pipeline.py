import os
import re
import uuid
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import open3d as o3d
import torch
import trimesh
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


class ImageTo3DService:
    """Dịch vụ chuyển một ảnh 2D thành depth map, point cloud và mesh 3D."""

    def __init__(self, model_id: str = "depth-anything/Depth-Anything-V2-Small-hf", output_dir: str = "outputs"):
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_side = int(os.getenv("MAX_IMAGE_SIDE", "768"))
        self.poisson_depth = int(os.getenv("POISSON_DEPTH", "8"))
        self.target_triangles = int(os.getenv("TARGET_TRIANGLES", "60000"))
        self.voxel_size = float(os.getenv("VOXEL_SIZE", "0.005"))

        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.model_id).to(self.device)
        if self.device == "cuda":
            self.model = self.model.half()
        self.model.eval()

    def estimate_depth(self, image_pil: Image.Image) -> np.ndarray:
        """Ước lượng depth map và chuẩn hóa về [0, 1]."""
        inputs = self.processor(images=image_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            if self.device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)

        pred_depth = outputs.predicted_depth
        pred_depth = torch.nn.functional.interpolate(
            pred_depth.unsqueeze(1),
            size=image_pil.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth = pred_depth.detach().cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth.astype(np.float32)

    def depth_to_point_cloud(self, image_rgb: np.ndarray, depth_norm: np.ndarray) -> o3d.geometry.PointCloud:
        """Tạo point cloud bằng Open3D từ RGB + depth."""
        h, w = depth_norm.shape
        depth_for_3d = 1.0 - depth_norm
        depth_u16 = np.clip(depth_for_3d * 1200.0, 1, 65535).astype(np.uint16)

        color_o3d = o3d.geometry.Image(image_rgb)
        depth_o3d = o3d.geometry.Image(depth_u16)

        fx = fy = max(w, h) * 1.2
        cx = w / 2.0
        cy = h / 2.0
        intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=1000.0,
            depth_trunc=5.0,
            convert_rgb_to_intensity=False,
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
        return pcd

    def point_cloud_to_mesh(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
        """Tái tạo mesh từ point cloud bằng Poisson reconstruction."""
        if len(pcd.points) < 100:
            raise ValueError("Point cloud quá ít điểm để dựng mesh. Hãy thử ảnh rõ hơn hoặc giảm VOXEL_SIZE.")

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(10)

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=self.poisson_depth)
        densities = np.asarray(densities)
        threshold = np.quantile(densities, 0.05)
        mesh.remove_vertices_by_mask(densities < threshold)

        bbox = pcd.get_axis_aligned_bounding_box()
        mesh = mesh.crop(bbox)

        if len(mesh.triangles) > self.target_triangles:
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=self.target_triangles)

        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.compute_vertex_normals()
        mesh = self.colorize_mesh_from_point_cloud(mesh, pcd)
        return mesh

    def colorize_mesh_from_point_cloud(
        self,
        mesh: o3d.geometry.TriangleMesh,
        pcd: o3d.geometry.PointCloud,
    ) -> o3d.geometry.TriangleMesh:
        """Gán màu vertex của mesh theo điểm point cloud gần nhất để GLB preview dễ nhìn hơn."""
        if not pcd.has_colors() or len(mesh.vertices) == 0:
            return mesh

        kdtree = o3d.geometry.KDTreeFlann(pcd)
        pcd_colors = np.asarray(pcd.colors)
        vertex_colors = []
        for vertex in np.asarray(mesh.vertices):
            _, idx, _ = kdtree.search_knn_vector_3d(vertex, 1)
            vertex_colors.append(pcd_colors[idx[0]] if idx else [0.8, 0.8, 0.8])
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(vertex_colors))
        return mesh

    def open_and_resize_image(self, image_path: str) -> Image.Image:
        """Mở ảnh và resize để phù hợp RAM/VRAM giới hạn của Colab free."""
        image_pil = Image.open(image_path).convert("RGB")
        w, h = image_pil.size
        scale = min(self.max_side / max(w, h), 1.0)
        if scale < 1.0:
            image_pil = image_pil.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        return image_pil

    @staticmethod
    def safe_stem(image_path: str) -> str:
        stem = Path(image_path).stem
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
        return stem or "image"

    def run(self, image_path: str) -> Dict[str, str]:
        """Chạy full pipeline và lưu kết quả ra thư mục outputs."""
        image_pil = self.open_and_resize_image(image_path)

        image_rgb = np.array(image_pil)
        depth_norm = self.estimate_depth(image_pil)

        pcd = self.depth_to_point_cloud(image_rgb, depth_norm)
        mesh = self.point_cloud_to_mesh(pcd)

        stem = self.safe_stem(image_path)
        run_id = uuid.uuid4().hex[:8]
        out_dir = self.output_dir / f"{stem}_{run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        depth_png = out_dir / "depth.png"
        ply_path = out_dir / "point_cloud.ply"
        obj_path = out_dir / "mesh.obj"
        glb_path = out_dir / "mesh.glb"

        depth_vis = (depth_norm * 255).astype(np.uint8)
        cv2.imwrite(str(depth_png), depth_vis)
        o3d.io.write_point_cloud(str(ply_path), pcd)
        o3d.io.write_triangle_mesh(str(obj_path), mesh, write_triangle_uvs=False)

        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        normals = np.asarray(mesh.vertex_normals) if len(mesh.vertex_normals) > 0 else None
        colors = None
        if mesh.has_vertex_colors():
            colors = (np.asarray(mesh.vertex_colors) * 255).clip(0, 255).astype(np.uint8)
        tm = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_normals=normals,
            vertex_colors=colors,
            process=False,
        )
        tm.export(str(glb_path))

        return {
            "job_id": out_dir.name,
            "depth_png": str(depth_png),
            "ply": str(ply_path),
            "obj": str(obj_path),
            "glb": str(glb_path),
        }
