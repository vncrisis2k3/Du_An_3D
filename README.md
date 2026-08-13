# 2D Image to 3D - Colab + FastAPI

Ứng dụng demo chuyển 1 ảnh 2D JPG/PNG thành depth map, point cloud và mesh 3D. Dự án có hai phần:

- `colab/image_to_3d_colab.ipynb`: notebook chạy thử trên Google Colab GPU miễn phí.
- `colab/triposr_360_colab.ipynb`: notebook tạo object 360 đẹp hơn bằng TripoSR.
- `colab/all_in_one_2d_to_3d_colab.ipynb`: notebook gộp cả TripoSR 360 và Depth Anything trong một file.
- `webapp/`: FastAPI backend + HTML/JS frontend dùng `model-viewer` để upload, preview và tải file 3D.

Hướng dẫn cấu hình Colab theo từng cell từ A-Z nằm ở: `COLAB_A_Z.md`.

## Chọn pipeline

- Muốn **360 độ đẹp cho vật thể đơn**: dùng `TripoSR`. Đây là chế độ mặc định trên web.
- Muốn **demo nhanh cho phong cảnh/relief/depth từ một góc nhìn**: dùng Depth Anything V2 + Open3D.

Pipeline depth nhanh:

1. Nhận ảnh đầu vào và resize về kích thước an toàn cho RAM/VRAM.
2. Ước lượng depth bằng `depth-anything/Depth-Anything-V2-Small-hf`.
3. Dựng point cloud có màu bằng Open3D từ RGB + depth.
4. Tái tạo mesh bằng Poisson reconstruction, dọn mesh và giảm số tam giác.
5. Xuất `depth.png`, `point_cloud.ply`, `mesh.obj`, `mesh.glb`.

Pipeline 360 đẹp:

1. Tách nền và căn object bằng TripoSR.
2. Sinh representation 3D từ một ảnh đơn.
3. Extract mesh bằng marching cubes.
4. Xuất `mesh.glb`, rồi convert thêm `mesh.obj` và `mesh.ply`.
5. Preview `mesh.glb` bằng Google `<model-viewer>`.

## Chạy trên Google Colab

Nếu muốn dùng một notebook duy nhất, mở:

```text
colab/all_in_one_2d_to_3d_colab.ipynb
```

Trong cell cấu hình, chọn:

```python
PIPELINE = 'triposr_360'
```

hoặc:

```python
PIPELINE = 'depth_relief'
```

Nếu mục tiêu là 360 đẹp và muốn notebook tách riêng, mở:

```text
colab/triposr_360_colab.ipynb
```

Nếu muốn bản depth nhanh, mở:

```text
colab/image_to_3d_colab.ipynb
```

Trong Colab, chọn runtime GPU rồi chạy lần lượt từ cell đầu.

Notebook depth có sẵn:

- Cell cài thư viện.
- Cell tải pretrained depth model.
- Demo upload ảnh bằng `google.colab.files.upload()`.
- Preview ảnh gốc, depth map và GLB ngay trong Colab.
- Cell tùy chọn mở FastAPI + HTML upload qua ngrok để có public URL demo.

Nếu Colab báo thiếu RAM/VRAM, giảm trong notebook:

```python
MAX_IMAGE_SIDE = 512
POISSON_DEPTH = 7
TARGET_TRIANGLES = 30000
VOXEL_SIZE = 0.008
```

Với notebook TripoSR, nếu T4 bị thiếu VRAM, giảm:

```bash
--mc-resolution 192 --chunk-size 2048
```

## Chạy web app độc lập

```bash
cd webapp
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Mở trình duyệt tại:

```text
http://localhost:8000
```

Nếu máy có GPU, hãy cài bản `torch` phù hợp CUDA trước hoặc thay dòng `torch` trong `requirements.txt` theo môi trường của bạn.

### Bật chế độ 360 TripoSR cho web

TripoSR là repo riêng, nên cần cài thêm:

```bash
git clone https://github.com/VAST-AI-Research/TripoSR.git
cd TripoSR
pip install -r requirements.txt
```

Sau đó chạy web app và trỏ `TRIPOSR_DIR` tới thư mục TripoSR:

```powershell
cd webapp
pip install -r requirements.txt
pip install -r requirements-triposr.txt
$env:TRIPOSR_DIR="D:\path\to\TripoSR"
$env:TRIPOSR_MC_RESOLUTION="256"
$env:TRIPOSR_CHUNK_SIZE="4096"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Trên giao diện, chọn chế độ `360 object đẹp (TripoSR)`.

## API

Endpoint:

```text
POST /api/generate
```

Form-data:

```text
file: JPG hoặc PNG
mode: triposr hoặc depth
```

Response:

```json
{
  "message": "Tao mo hinh 3D thanh cong",
  "job_id": "input_ab12cd34",
  "device": "cuda",
  "model": "TripoSR",
  "mode": "triposr",
  "preview_glb_url": "/outputs/input_ab12cd34/mesh.glb",
  "download": {
    "ply": "/outputs/input_ab12cd34/mesh.ply",
    "obj": "/outputs/input_ab12cd34/mesh.obj",
    "glb": "/outputs/input_ab12cd34/mesh.glb"
  }
}
```

## Cấu hình model và hiệu năng

Trong `webapp`, có thể đổi model hoặc giảm tải bằng biến môi trường:

```powershell
$env:DEPTH_MODEL_ID="depth-anything/Depth-Anything-V2-Base-hf"
$env:MAX_IMAGE_SIDE="512"
$env:POISSON_DEPTH="7"
$env:TARGET_TRIANGLES="30000"
$env:VOXEL_SIZE="0.008"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Gợi ý thay thế:

- Nhanh, nhẹ VRAM: `depth-anything/Depth-Anything-V2-Small-hf`.
- Depth tốt hơn nhưng nặng hơn: Depth Anything V2 Base/Large hoặc MiDaS.
- Vật thể 360 hoàn chỉnh hơn: dùng TripoSR trong repo này, hoặc nâng tiếp sang Stable Fast 3D, Zero123++, Shap-E hay một model image-to-3D chuyên dụng.

## Giới hạn cần biết

TripoSR tạo 360 tốt nhất với một vật thể đơn, rõ biên, nền đơn giản, ánh sáng đều, góc nhìn trước hoặc 3/4. Mặt sau vẫn là phần model suy luận từ một ảnh, nên các chi tiết khuất, vật thể trong suốt, phản chiếu mạnh hoặc cấu trúc quá mảnh có thể kém chính xác.

Pipeline depth -> mesh chỉ suy ra hình học từ một góc nhìn, nên mặt sau của vật thể không thể chính xác như model 3D sinh đa góc. Với ảnh chân dung/vật thể đơn, kết quả tốt nhất thường là dạng relief/mesh một mặt hoặc point cloud; muốn 360 độ đẹp nên dùng chế độ TripoSR.
