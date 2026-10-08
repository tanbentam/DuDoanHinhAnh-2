"""
KHUNG DEMO (Gradio) — các nhóm chỉ cần cài đặt hàm `load_predictor()`.

Giao diện gồm 2 tab:
  1. "Ảnh của bạn": tải ảnh / chụp webcam -> hiển thị ảnh 32x32 mà mô hình thực sự
     nhìn thấy (phóng to) + top-3 lớp kèm xác suất.
  2. "Ảnh validation": lấy ngẫu nhiên ảnh từ tập validation, so sánh nhãn thật và dự đoán.

Chạy:
    pip install gradio
    python demo_app.py              # dùng mô hình của nhóm (sau khi cài load_predictor)
    python demo_app.py --dummy      # xem thử giao diện với dự đoán ngẫu nhiên
"""
import argparse
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from photo_to_input import photo_to_input

DATA_DIR = Path(__file__).parent / "data"

# predict_proba: nhận uint8 (N, 32, 32, 3) -> trả về xác suất float (N, 10)
Predictor = Callable[[np.ndarray], np.ndarray]


def load_predictor() -> Predictor:
    """TODO (nhóm cài đặt): nạp checkpoint và trả về hàm predict_proba.

    Ví dụ:
        model = MyMLP(...); model.load_state_dict(torch.load("best.pt")); model.eval()
        def predict_proba(images_uint8):
            x = preprocess(images_uint8)            # CÙNG tiền xử lý như lúc huấn luyện!
            with torch.no_grad():
                return torch.softmax(model(x), 1).numpy()
        return predict_proba
    """
    raise NotImplementedError("Hãy cài đặt load_predictor() trong demo_app.py")


def dummy_predictor(images: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(int(images.sum()) % 2**32)
    p = rng.random((len(images), 10)) ** 4
    return p / p.sum(1, keepdims=True)


def enlarge(x: np.ndarray, scale: int = 8) -> np.ndarray:
    """Phóng to ảnh 32x32 bằng nearest-neighbor để thấy rõ từng pixel."""
    return np.asarray(Image.fromarray(x).resize((32 * scale, 32 * scale), Image.NEAREST))


def build_app(predict_proba: Predictor, class_names, val_images=None, val_labels=None,
              title="MLP Challenge · CIFAR-10"):
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
        gr.Markdown(f"# {title}\nMô hình chỉ nhìn thấy ảnh **32×32 pixel** — "
                    "đây là ảnh trong ô **Mô hình nhìn thấy**.")
        with gr.Tab("Ảnh của bạn"):
            with gr.Row():
                inp = gr.Image(type="pil", sources=["upload", "webcam", "clipboard"],
                               label="Ảnh gốc")
                seen = gr.Image(label="Mô hình nhìn thấy (32×32)", interactive=False)
                out = gr.Label(num_top_classes=3, label="Top-3")
            inp.change(classify, inputs=inp, outputs=[seen, out])
        if val_images is not None:
            with gr.Tab("Ảnh validation"):
                btn = gr.Button("Lấy ảnh ngẫu nhiên")
                with gr.Row():
                    v_img = gr.Image(label="Ảnh 32×32", interactive=False)
                    v_out = gr.Label(num_top_classes=3, label="Top-3")
                v_txt = gr.Markdown()
                btn.click(random_val, outputs=[v_img, v_out, v_txt])
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dummy", action="store_true", help="Dự đoán ngẫu nhiên để thử giao diện.")
    ap.add_argument("--n-val", type=int, default=5000,
                    help="Lấy N ảnh cuối của train.npz làm ảnh validation (khớp cách nhóm chia).")
    args = ap.parse_args()

    train = np.load(DATA_DIR / "train.npz")
    class_names = [str(c) for c in train["class_names"]]
    val_images, val_labels = train["images"][-args.n_val:], train["labels"][-args.n_val:]
    predictor = dummy_predictor if args.dummy else load_predictor()
    build_app(predictor, class_names, val_images, val_labels).launch()


if __name__ == "__main__":
    main()
