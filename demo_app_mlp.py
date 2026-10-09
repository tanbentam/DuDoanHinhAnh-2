"""
GIAO DIỆN DEMO (Gradio) — Dành riêng cho Mô hình Deep Residual MLP.
Kế thừa hoàn toàn từ demo_app.py của giảng viên nhưng đã cài đặt sẵn load_predictor()
kết nối trực tiếp với checkpoint 'best_mlp_checkpoint.pt'.

Chạy:
    python demo_app_mlp.py              # chạy với mô hình đã huấn luyện
    python demo_app_mlp.py --dummy      # chạy thử nghiệm ngẫu nhiên nếu chưa train
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import argparse
from pathlib import Path
from typing import Callable
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from photo_to_input import photo_to_input

DATA_DIR = Path(__file__).parent / "data"
CHECKPOINT_PATH = Path(__file__).parent / "best_mlp_checkpoint.pt"

Predictor = Callable[[np.ndarray], np.ndarray]


# Định nghĩa kiến trúc khớp với ChampionDeepMLP trong MLP_Challenge_Experiments.ipynb
class FinalResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.25):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.act1 = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.act2 = nn.GELU()
    def forward(self, x):
        return self.act2(self.bn2(self.fc2(self.drop(self.act1(self.bn1(self.fc1(x)))))) + x)


class ChampionDeepMLP(nn.Module):
    def __init__(self, in_features=3072, hidden_dim=1024, num_classes=10, num_blocks=5, dropout=0.25):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        self.blocks = nn.ModuleList([FinalResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        x = self.in_proj(x)
        for b in self.blocks:
            x = b(x)
        return self.classifier(x)


def load_predictor() -> Predictor:
    """Nạp checkpoint 'best_mlp_checkpoint.pt' và trả về hàm dự đoán xác suất."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChampionDeepMLP(hidden_dim=1024, num_blocks=5).to(device)
    
    if CHECKPOINT_PATH.exists():
        print(f"[*] Đang nạp checkpoint mô hình: {CHECKPOINT_PATH}")
        weights = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(weights)
    else:
        print(f"[!] Chưa tìm thấy '{CHECKPOINT_PATH}'. Đang dùng trọng số khởi tạo (chưa train).")
        print("    Vui lòng chạy notebook 'MLP_Challenge_Experiments.ipynb' để sinh checkpoint tốt nhất!")

    model.eval()

    def predict_proba(images_uint8: np.ndarray) -> np.ndarray:
        # Chuẩn hóa về [0, 1] và làm phẳng sang vector 3072 chiều cho MLP
        x_norm = images_uint8.reshape(len(images_uint8), -1).astype(np.float32) / 255.0
        x_tensor = torch.from_numpy(x_norm).to(device)
        with torch.no_grad():
            logits = model(x_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    return predict_proba


def dummy_predictor(images: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(int(images.sum()) % 2**32)
    p = rng.random((len(images), 10)) ** 4
    return p / p.sum(1, keepdims=True)


def enlarge(x: np.ndarray, scale: int = 8) -> np.ndarray:
    return np.asarray(Image.fromarray(x).resize((32 * scale, 32 * scale), Image.NEAREST))


def build_app(predict_proba: Predictor, class_names, val_images=None, val_labels=None,
              title="MLP Challenge · CIFAR-10 (Deep Residual MLP)"):
    import gradio as gr

    def classify(img):
        if img is None:
            return None, None
        x = photo_to_input(img)
        p = predict_proba(x[None])[0]
        return enlarge(x), {class_names[i]: float(p[i]) for i in range(len(class_names))}

    def random_val():
        i = np.random.randint(len(val_images))
        x = val_images[i]
        p = predict_proba(x[None])[0]
        truth = class_names[val_labels[i]]
        pred = class_names[int(p.argmax())]
        verdict = "✅ Đúng" if truth == pred else "❌ Sai"
        return (enlarge(x), {class_names[k]: float(p[k]) for k in range(len(class_names))},
                f"Nhãn thật: **{truth}** · Dự đoán: **{pred}** · {verdict}")

    with gr.Blocks(title=title) as app:
        gr.Markdown(f"# {title}\nMô hình **Deep Residual MLP** nhận dạng ảnh **32×32 pixel** trên vector phẳng 3072 chiều.")
        with gr.Tab("Ảnh của bạn"):
            with gr.Row():
                inp = gr.Image(type="pil", sources=["upload", "webcam", "clipboard"], label="Ảnh gốc")
                seen = gr.Image(label="Mô hình nhìn thấy (32×32)", interactive=False)
                out = gr.Label(num_top_classes=3, label="Top-3 Dự đoán")
            inp.change(classify, inputs=inp, outputs=[seen, out])
        if val_images is not None:
            with gr.Tab("Ảnh validation"):
                btn = gr.Button("Lấy ảnh ngẫu nhiên")
                with gr.Row():
                    v_img = gr.Image(label="Ảnh 32×32", interactive=False)
                    v_out = gr.Label(num_top_classes=3, label="Top-3 Dự đoán")
                v_txt = gr.Markdown()
                btn.click(random_val, outputs=[v_img, v_out, v_txt])
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dummy", action="store_true", help="Dự đoán ngẫu nhiên để thử giao diện.")
    ap.add_argument("--n-val", type=int, default=5000, help="Số ảnh validation.")
    args = ap.parse_args()

    train = np.load(DATA_DIR / "train.npz")
    class_names = [str(c) for c in train["class_names"]]
    val_images, val_labels = train["images"][-args.n_val:], train["labels"][-args.n_val:]
    predictor = dummy_predictor if args.dummy else load_predictor()
    build_app(predictor, class_names, val_images, val_labels).launch()


if __name__ == "__main__":
    main()
