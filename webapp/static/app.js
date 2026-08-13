const form = document.getElementById("uploadForm");
const fileInput = document.getElementById("imageInput");
const statusBox = document.getElementById("status");
const viewer = document.getElementById("viewer");
const linksBox = document.getElementById("downloadLinks");
const submitBtn = document.getElementById("submitBtn");
const sourcePreview = document.getElementById("sourcePreview");
const modeSelect = document.getElementById("modeSelect");

function setStatus(text) {
  statusBox.textContent = text;
}

function createDownloadLink(label, href) {
  const a = document.createElement("a");
  a.href = href;
  a.textContent = `Tải ${label}`;
  a.target = "_blank";
  a.rel = "noopener noreferrer";
  return a;
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files && fileInput.files[0];
  if (!file) {
    sourcePreview.removeAttribute("src");
    return;
  }
  sourcePreview.src = URL.createObjectURL(file);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!fileInput.files || fileInput.files.length === 0) {
    setStatus("Bạn chưa chọn file ảnh.");
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  formData.append("mode", modeSelect.value);

  submitBtn.disabled = true;
  const label = modeSelect.value === "triposr" ? "360 bằng TripoSR" : "depth nhanh";
  setStatus(`Đang xử lý ${label}, vui lòng đợi...`);
  linksBox.innerHTML = "";

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Không thể tạo mô hình 3D");
    }

    viewer.src = `${data.preview_glb_url}?t=${Date.now()}`;
    setStatus(`Đã tạo xong mô hình 3D bằng ${data.model} trên ${data.device}.`);

    linksBox.appendChild(createDownloadLink("GLB", data.download.glb));
    linksBox.appendChild(createDownloadLink("OBJ", data.download.obj));
    linksBox.appendChild(createDownloadLink("PLY", data.download.ply));
    if (data.download.depth_png) {
      linksBox.appendChild(createDownloadLink("Depth PNG", data.download.depth_png));
    }
  } catch (error) {
    setStatus(`Loi: ${error.message}`);
  } finally {
    submitBtn.disabled = false;
  }
});
