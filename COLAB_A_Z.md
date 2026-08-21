# Cấu Hình Colab Từ A-Z

Tài liệu này hướng dẫn chạy toàn bộ các cell trên Google Colab cho hai mục tiêu:

- **360 độ đẹp cho vật thể đơn**: dùng `colab/triposr_360_colab.ipynb`.
- **Depth/point cloud/relief nhanh**: dùng `colab/image_to_3d_colab.ipynb`.
- **Một file gộp cả hai chế độ**: dùng `colab/all_in_one_2d_to_3d_colab.ipynb`.

Nếu mục tiêu chính của bạn là mô hình xoay 360 đẹp, hãy bắt đầu với notebook TripoSR.

## Notebook Gộp Một File

File nên dùng nếu bạn muốn đơn giản hóa:

```text
colab/all_in_one_2d_to_3d_colab.ipynb
```

Trong `Cell 2`, chọn pipeline:

```python
PIPELINE = 'triposr_360'
```

hoặc:

```python
PIPELINE = 'depth_relief'
```

Các cell còn lại sẽ tự chạy đúng nhánh. Cell TripoSR sẽ tự bỏ qua nếu bạn chọn depth, và cell depth sẽ tự bỏ qua nếu bạn chọn TripoSR.

Preset TripoSR đẹp hơn trong notebook All-in-One:

```python
TRIPOSR_MC_RESOLUTION = 320
TRIPOSR_CHUNK_SIZE = 4096
TRIPOSR_FOREGROUND_RATIO = 0.8
TRIPOSR_BAKE_TEXTURE = True
TRIPOSR_TEXTURE_RESOLUTION = 2048
```

`TRIPOSR_BAKE_TEXTURE = True` giúp kết quả bớt giống tượng gỗ/nhựa vì TripoSR sẽ bake texture atlas thay vì chỉ dùng vertex color.

## A. Chuẩn Bị Trên Google Colab

1. Mở Google Colab: `https://colab.research.google.com`.
2. Chọn `File > Upload notebook`.
3. Upload một trong hai file:
   - `colab/triposr_360_colab.ipynb`
   - `colab/image_to_3d_colab.ipynb`
4. Chọn GPU:
   - Vào `Runtime > Change runtime type`.
   - `Hardware accelerator`: chọn `T4 GPU` hoặc `GPU`.
   - Bấm `Save`.
5. Kiểm tra góc trên phải có RAM/Disk/GPU runtime.

## B. Notebook 360 Đẹp Bằng TripoSR

File cần mở:

```text
colab/triposr_360_colab.ipynb
```

Đây là pipeline nên dùng khi ảnh là **một vật thể đơn** như ghế, giày, balo, tượng, nhân vật, sản phẩm.

### Cell 1 - Giới Thiệu

Cell markdown giải thích mục tiêu notebook.

Không cần chỉnh gì.

### Cell 2 - Kiểm Tra GPU

```python
!nvidia-smi
```

Kết quả mong muốn:

- Có dòng `Tesla T4`, `L4`, `A100`, hoặc GPU khác.
- Nếu báo không có `nvidia-smi`, hãy vào `Runtime > Change runtime type > GPU`.

### Cell 3 - Cài Đặt TripoSR

Cell này cài PyTorch, clone repo TripoSR và cài requirements:

```python
!pip -q install --upgrade pip
!pip -q install "setuptools<82" wheel "jedi>=0.16"
!pip -q install torch torchvision --index-url https://download.pytorch.org/whl/cu124
!git clone https://github.com/VAST-AI-Research/TripoSR.git /content/TripoSR
%cd /content/TripoSR
!pip -q install -r requirements.txt
!pip -q uninstall -y cupy-cuda12x cupy
!pip -q install --force-reinstall --no-cache-dir "numpy==1.26.4" "cupy-cuda12x==13.6.0" onnxruntime trimesh
```

Nếu chạy lại notebook lần hai và `git clone` báo thư mục đã tồn tại, dùng cell thay thế:

```python
!rm -rf /content/TripoSR
!git clone https://github.com/VAST-AI-Research/TripoSR.git /content/TripoSR
%cd /content/TripoSR
!pip -q install -r requirements.txt
```

Nếu lỗi `torchmcubes`, chạy:

```python
!pip uninstall -y torchmcubes
!pip install git+https://github.com/tatsy/torchmcubes.git
```

Nếu lỗi:

```text
ModuleNotFoundError: No module named 'onnxruntime'
```

Chạy cell sửa nhanh này:

```python
!pip -q install onnxruntime
```

Sau đó chạy lại cell TripoSR inference, tức cell có lệnh:

```python
!python run.py "$input_path" ...
```

Nếu lỗi CuPy/NumPy:

```text
Failed to import CuPy
Original error:
ImportError: numpy.core.multiarray failed to import
```

Nguyên nhân thường là Colab có sẵn `cupy-cuda12x` mới, nhưng TripoSR/rembg kéo `numpy==1.26.4`. Cài lại cặp tương thích:

```python
!pip -q uninstall -y cupy-cuda12x cupy
!pip -q install --force-reinstall --no-cache-dir "numpy==1.26.4" "cupy-cuda12x==13.6.0" onnxruntime trimesh
```

Sau đó nên chọn:

```text
Runtime > Restart runtime
```

Rồi chạy lại từ Cell 1 đến Cell 6A. CuPy `13.6.0` có wheel cho Python 3.12 và CUDA 12.x, hợp với môi trường Colab mới hơn.

Sau Cell 3, có thể kiểm tra nhanh:

```python
import numpy
import cupy
import onnxruntime

print('numpy:', numpy.__version__)
print('cupy:', cupy.__version__)
print('onnxruntime:', onnxruntime.__version__)
```

Nếu import được cả 3 thư viện, các cảnh báo dependency conflict trước đó có thể bỏ qua.

Nếu thấy log kiểu `pip's dependency resolver does not currently take into account...`, thường đó là cảnh báo do Colab đã cài sẵn nhiều package khác phiên bản. Với TripoSR, các cảnh báo về `huggingface-hub`, `transformers`, `websockets`, `numpy`, `diffusers`, `jax`, `opencv`, `datasets`, `sentence-transformers`, hoặc `typer does not provide the extra 'all'` thường có thể bỏ qua nếu cell vẫn chạy tiếp và không có traceback ở cuối.

Nếu thấy cảnh báo:

```text
torch ... requires setuptools<82, but you have setuptools 84.0.0
```

Chạy cell sửa nhanh này rồi chạy lại cell cài đặt:

```python
!pip -q install "setuptools<82" "jedi>=0.16"
```

Nếu cell cài đặt đã chạy xong và cell inference vẫn chạy được, bạn không cần xử lý thêm.

### Cell 4 - Upload Ảnh

Cell này mở hộp thoại upload:

```python
uploaded = files.upload()
```

Ảnh nên có:

- Một vật thể chính, chiếm phần lớn khung hình.
- Nền đơn giản.
- Ánh sáng đều.
- Ảnh nét, không motion blur.
- Góc chụp trước hoặc 3/4.

Ảnh không nên dùng:

- Phong cảnh.
- Nhiều vật thể lẫn nhau.
- Vật thể trong suốt hoặc quá phản chiếu.
- Vật thể quá mảnh như dây điện, tóc rối, lưới.

### Cell 5 - Chạy TripoSR

Cell mặc định:

```python
!python run.py "$image_path" --output-dir "$output_dir" --model-save-format glb --mc-resolution 256 --chunk-size 4096 --foreground-ratio 0.85
```

Ý nghĩa cấu hình:

- `--model-save-format glb`: xuất file GLB để preview web dễ nhất.
- `--mc-resolution 256`: chất lượng mesh tốt, hợp với T4 nếu đủ VRAM.
- `--chunk-size 4096`: giảm VRAM khi extract/render.
- `--foreground-ratio 0.85`: object chiếm 85% khung sau khi tách nền.
- `--bake-texture`: tạo texture atlas, nhìn giống vật liệu ảnh gốc hơn so với vertex color.
- `--texture-resolution 2048`: độ phân giải texture.

Trong All-in-One, nếu bật:

```python
TRIPOSR_BAKE_TEXTURE = True
```

Cell 6A sẽ export OBJ/textured trước, sau đó convert thêm sang GLB để preview.

Nếu bị thiếu VRAM, đổi thành:

```python
!python run.py "$image_path" --output-dir "$output_dir" --model-save-format glb --mc-resolution 192 --chunk-size 2048 --foreground-ratio 0.85
```

Nếu vẫn thiếu VRAM:

```python
!python run.py "$image_path" --output-dir "$output_dir" --model-save-format glb --mc-resolution 128 --chunk-size 1024 --foreground-ratio 0.8
```

Nếu object bị crop sát quá, giảm:

```text
--foreground-ratio 0.75
```

Nếu object quá nhỏ trong preview, tăng:

```text
--foreground-ratio 0.9
```

Nếu kết quả giống tượng gỗ, ít chi tiết:

```python
TRIPOSR_MC_RESOLUTION = 320
TRIPOSR_BAKE_TEXTURE = True
TRIPOSR_TEXTURE_RESOLUTION = 2048
TRIPOSR_FOREGROUND_RATIO = 0.8
```

Nếu T4 hết VRAM với cấu hình trên:

```python
TRIPOSR_MC_RESOLUTION = 256
TRIPOSR_TEXTURE_RESOLUTION = 1024
TRIPOSR_CHUNK_SIZE = 2048
```

Nếu vẫn hết VRAM:

```python
TRIPOSR_MC_RESOLUTION = 192
TRIPOSR_BAKE_TEXTURE = False
TRIPOSR_CHUNK_SIZE = 2048
```

Nếu lỗi:

```text
No such file or directory: '/content/TripoSR'
python3: can't open file '/content/all_in_one_2d_to_3d/inputs/run.py'
```

Nguyên nhân là runtime không còn thư mục TripoSR, hoặc Colab đang đứng ở thư mục input nên lệnh `python run.py` bị chạy sai chỗ. Với notebook All-in-One bản mới, Cell 6A đã dùng đường dẫn tuyệt đối `/content/TripoSR/run.py` để tránh lỗi này.

Cách sửa nhanh:

1. Chạy lại Cell 3 để clone/cài TripoSR.
2. Chạy lại Cell 5 nếu muốn upload ảnh mới.
3. Chạy Cell 6A.

Không tin output cũ nếu cell có lỗi `run.py`; hãy chạy lại Cell 6A sau khi Cell 3 thành công.

### Cell 6 - Preview GLB

Cell này nhúng file `mesh.glb` vào `model-viewer`.

Bạn có thể:

- Kéo chuột để xoay 360.
- Cuộn để zoom.
- Kiểm tra mặt trước, mặt sau, mặt bên.

Nếu preview trắng:

1. Kiểm tra cell trước có tạo được `mesh.glb` không.
2. Chạy lại cell preview.
3. Nếu file GLB quá nặng, giảm `--mc-resolution`.

### Cell 7 - Xuất OBJ/PLY Và Tải Về

Cell này convert GLB sang OBJ/PLY bằng `trimesh`, rồi tải về:

```python
files.download(str(glb_path))
files.download(str(obj_path))
files.download(str(ply_path))
```

File nên dùng:

- `.glb`: tốt nhất cho web, Three.js, model-viewer, Blender.
- `.obj`: dễ import vào nhiều phần mềm 3D.
- `.ply`: tốt cho point/mesh processing.

## C. Notebook Depth Nhanh Bằng Depth Anything V2

File cần mở:

```text
colab/image_to_3d_colab.ipynb
```

Pipeline này phù hợp với:

- Phong cảnh.
- Ảnh phòng/không gian.
- Demo depth map.
- Relief mesh từ một góc nhìn.

Pipeline này không tạo mặt sau 360 chính xác.

### Cell 1 - Giới Thiệu

Không cần chỉnh gì.

### Cell 2 - Cài Thư Viện

Cell cài PyTorch, Transformers, Open3D, Trimesh, FastAPI:

```python
!pip -q install --upgrade pip
!pip -q install "setuptools<82" wheel "jedi>=0.16"
!pip -q install torch torchvision --index-url https://download.pytorch.org/whl/cu124
!pip -q install transformers accelerate timm opencv-python pillow matplotlib numpy open3d trimesh fastapi uvicorn pyngrok python-multipart nest_asyncio
```

Nếu cài `open3d` lâu, chờ thêm. Colab có thể mất vài phút.

### Cell 3 - Import Và Cấu Hình Chung

Thông số quan trọng:

```python
MAX_IMAGE_SIDE = 768
POISSON_DEPTH = 8
TARGET_TRIANGLES = 60000
VOXEL_SIZE = 0.005
```

Cấu hình nhẹ cho T4 yếu/RAM thấp:

```python
MAX_IMAGE_SIDE = 512
POISSON_DEPTH = 7
TARGET_TRIANGLES = 30000
VOXEL_SIZE = 0.008
```

Cấu hình đẹp hơn nhưng nặng hơn:

```python
MAX_IMAGE_SIDE = 1024
POISSON_DEPTH = 9
TARGET_TRIANGLES = 100000
VOXEL_SIZE = 0.004
```

### Cell 4 - Load Model Depth

Mặc định:

```python
MODEL_ID = 'depth-anything/Depth-Anything-V2-Small-hf'
```

Gợi ý đổi model:

```python
MODEL_ID = 'depth-anything/Depth-Anything-V2-Base-hf'
```

Small nhanh và nhẹ hơn. Base thường depth tốt hơn nhưng dùng nhiều VRAM hơn.

### Cell 5 - Hàm Xử Lý Chính

Cell này định nghĩa các hàm:

- `open_and_resize_image`
- `estimate_depth`
- `depth_to_point_cloud`
- `point_cloud_to_mesh`
- `save_outputs`
- `run_pipeline`

Không cần chỉnh nếu bạn chỉ muốn chạy demo.

Chỉ chỉnh khi muốn giảm/tăng chất lượng:

- Giảm `VOXEL_SIZE` để point cloud dày hơn.
- Tăng `POISSON_DEPTH` để mesh chi tiết hơn.
- Tăng `TARGET_TRIANGLES` để giữ nhiều tam giác hơn.

### Cell 6 - Upload Ảnh Và Chạy Pipeline

Cell này upload ảnh rồi chạy:

```python
image_rgb, depth_norm, pcd, mesh, artifacts = run_pipeline(str(input_path))
```

Kết quả in ra:

- Đường dẫn output.
- Số point trong point cloud.
- Số vertex/tam giác của mesh.

Nếu lỗi `Point cloud qua it diem`, thử:

```python
VOXEL_SIZE = 0.003
MAX_IMAGE_SIDE = 768
```

Rồi chạy lại từ cell định nghĩa hàm trở xuống.

Nếu lỗi khi upload:

```text
FileNotFoundError: [Errno 2] No such file or directory: 'ten_anh.png'
```

Nguyên nhân thường là Colab đang đứng trong thư mục làm việc đã bị xóa, hay gặp sau khi chạy lại cell cài TripoSR có lệnh `rm -rf /content/TripoSR`. Cách sửa nhanh cho All-in-One là chạy cell này trước Cell 5:

```python
import os
from pathlib import Path

INPUT_DIR = Path('/content/all_in_one_2d_to_3d/inputs')
INPUT_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(str(INPUT_DIR))
print('Upload dir:', INPUT_DIR)
```

Sau đó chạy lại Cell 5.

### Cell 7 - Hiển Thị Ảnh Gốc Và Depth Map

Cell này giúp kiểm tra depth có hợp lý không.

Depth map tốt thường có:

- Vật gần/sâu tách rõ.
- Biên vật thể không quá nhiễu.
- Không bị toàn đen hoặc toàn trắng.

### Cell 8 - Preview GLB

Preview mesh bằng `model-viewer`.

Với depth pipeline, mesh thường giống dạng địa hình/relief hơn là object 360 hoàn chỉnh.

### Cell 9 - Tải File Kết Quả

Tải:

- `mesh.glb`
- `mesh.obj`
- `point_cloud.ply`
- `depth.png`

### Cell 10-11 - API/Web Mini Qua Ngrok

Cell cuối tạo FastAPI và HTML upload chạy ngay trong Colab.

Nếu ngrok yêu cầu token:

1. Tạo tài khoản tại `https://ngrok.com`.
2. Lấy authtoken.
3. Thêm cell trước cell API:

```python
import os
os.environ['NGROK_AUTHTOKEN'] = 'PASTE_TOKEN_CUA_BAN'
```

Sau đó chạy cell API.

Khi thấy:

```text
Public URL: https://xxxx.ngrok-free.app
```

Mở URL đó để demo upload ảnh từ web.

## D. Chọn Cấu Hình Theo Máy

### T4 Free Colab - Khuyến Nghị

TripoSR:

```bash
--mc-resolution 192 --chunk-size 2048
```

Depth:

```python
MAX_IMAGE_SIDE = 512
POISSON_DEPTH = 7
TARGET_TRIANGLES = 30000
VOXEL_SIZE = 0.008
```

### T4 Free Colab - Chất Lượng Cao Hơn

TripoSR:

```bash
--mc-resolution 256 --chunk-size 4096
```

Depth:

```python
MAX_IMAGE_SIDE = 768
POISSON_DEPTH = 8
TARGET_TRIANGLES = 60000
VOXEL_SIZE = 0.005
```

### Colab Pro Hoặc GPU Mạnh

TripoSR:

```bash
--mc-resolution 320 --chunk-size 8192
```

Depth:

```python
MAX_IMAGE_SIDE = 1024
POISSON_DEPTH = 9
TARGET_TRIANGLES = 100000
VOXEL_SIZE = 0.004
```

## E. Checklist Khi Kết Quả 360 Chưa Đẹp

1. Dùng ảnh vật thể đơn, nền sạch.
2. Crop ảnh để object chiếm 60-85% khung hình.
3. Tránh ảnh quá tối, quá chói, hoặc bóng đổ gắt.
4. Dùng góc 3/4 thay vì chính diện nếu vật thể có hình khối phức tạp.
5. Tăng `--mc-resolution` nếu mesh thiếu chi tiết.
6. Giảm `--foreground-ratio` nếu object bị cắt mất viền.
7. Thử xóa nền trước bằng remove.bg, Photoshop, Canva hoặc rembg nếu nền quá rối.

## F. Lỗi Thường Gặp

### Không Có GPU

Vào:

```text
Runtime > Change runtime type > Hardware accelerator > GPU
```

### CUDA Out Of Memory

TripoSR:

```bash
--mc-resolution 128 --chunk-size 1024
```

Depth:

```python
MAX_IMAGE_SIDE = 512
POISSON_DEPTH = 7
TARGET_TRIANGLES = 30000
VOXEL_SIZE = 0.01
```

### Ngrok Không Chạy

Thêm token:

```python
import os
os.environ['NGROK_AUTHTOKEN'] = 'PASTE_TOKEN_CUA_BAN'
```

### Preview GLB Không Hiện

1. Kiểm tra file `mesh.glb` có tồn tại.
2. Chạy lại cell preview.
3. Giảm độ phân giải mesh nếu file quá nặng.

### Pip Báo Dependency Conflict

Ví dụ:

```text
diffusers requires huggingface-hub...
sentence-transformers requires transformers...
jax requires numpy...
```

Đây thường là cảnh báo của môi trường Colab có nhiều thư viện cài sẵn. Nếu dòng cuối không phải `ERROR` kèm traceback Python và cell vẫn hoàn tất, cứ chạy cell tiếp theo.

Cell sửa nhanh nên chạy khi gặp cảnh báo `setuptools`:

```python
!pip -q install "setuptools<82" "jedi>=0.16"
```

## G. Kết Luận Nên Dùng Gì

- Muốn sản phẩm demo đẹp, xoay 360: dùng `triposr_360_colab.ipynb`.
- Muốn nghiên cứu depth, point cloud, cảnh/phong cảnh: dùng `image_to_3d_colab.ipynb`.
- Muốn web thật: chạy `webapp` và bật `TRIPOSR_DIR` để dùng chế độ TripoSR.
