# BÀI TẬP — MẠNG NƠRON VÀ HỌC SÂU

## "MLP Challenge": Nhận dạng ảnh thật (CIFAR-10) bằng mạng nơron nhiều lớp


## 2. Bài toán

Cho một **ảnh chụp thật, màu, kích thước 32 × 32** (bộ dữ liệu CIFAR-10), dự đoán vật thể chính trong ảnh thuộc 1 trong **10 lớp**:

| Nhãn | Lớp | Nhãn | Lớp |
|---|---|---|---|
| 0 | airplane (máy bay) | 5 | dog (chó) |
| 1 | automobile (ô tô con) | 6 | frog (ếch) |
| 2 | bird (chim) | 7 | horse (ngựa) |
| 3 | cat (mèo) | 8 | ship (tàu thủy) |
| 4 | deer (hươu, nai) | 9 | truck (xe tải) |

**Độ đo đánh giá:** Accuracy (tỉ lệ dự đoán đúng) trên tập test ẩn nhãn.
---

## 3. Dữ liệu

Thư mục `data/` gồm:

| File | Nội dung |
|---|---|
| `train.npz` | `images`: uint8 (50000, 32, 32, 3) — `labels`: int (50000,) — `class_names` |
| `test.npz` | `images`: uint8 (10000, 32, 32, 3) — `ids`: int (10000,) — **không có nhãn** |
| `sample_submission.csv` | File nộp mẫu, đúng định dạng |

Đọc dữ liệu:

```python
import numpy as np
train = np.load("data/train.npz")
x, y = train["images"], train["labels"]          # (50000, 32, 32, 3) RGB, (50000,)
test = np.load("data/test.npz")
x_test, ids = test["images"], test["ids"]         # (10000, 32, 32, 3), (10000,)
x_flat = x.reshape(len(x), -1)                    # vector 3072 chiều cho MLP
```

**Định dạng file nộp** (CSV, đúng 10.000 dòng dữ liệu, mỗi `id` xuất hiện đúng 1 lần):

```
id,label
0,9
1,2
...
```


## 4. Tasks

### 4.1. Baseline
File `baseline.py` là một MLP 1 lớp ẩn (sigmoid, SGD thuần, LR cố định, không regularization)

```bash
pip install -r requirements.txt
python baseline.py          # sinh ra submissions/baseline.csv
```

### 4.2. Phần bắt buộc — Thực nghiệm có kiểm soát với 5 nhóm kỹ thuật

Với mỗi nhóm kỹ thuật dưới đây, nhóm phải **thiết kế thí nghiệm so sánh có kiểm soát** (giữ nguyên các yếu tố khác, cùng epoch, đánh giá trên **cùng tập validation**), vẽ đồ thị và **giải thích hiện tượng** bằng kiến thức trong bài giảng.

| # | Nhóm kỹ thuật | Thí nghiệm tối thiểu | Cần thể hiện |
|---|---|---|---|
| E1 | **Optimizer** | SGD vs SGD+Momentum vs RMSProp vs Adam vs AdamW | Đường loss/accuracy theo epoch; nhận xét tốc độ hội tụ và độ ổn định |
| E2 | **Khởi tạo trọng số** | Mạng sâu (≥ 10 lớp ẩn, *không* residual, *không* normalization): khởi tạo quá nhỏ / quá lớn / Xavier / He, với ReLU và tanh | Biểu đồ **độ lệch chuẩn activation (và gradient) theo từng lớp** khi khởi tạo; kết quả huấn luyện |
| E3 | **Lịch learning rate** | LR cố định (cao/thấp) vs Step decay vs Cosine vs Warmup + Cosine | Đồ thị LR theo step và loss tương ứng |
| E4 | **Regularization** | Mạng rộng không regularization (trên CIFAR-10 MLP overfit rất nhanh), sau đó thêm lần lượt: weight decay, dropout, data augmentation (crop + lật), early stopping | Đường train–val trước/sau; bảng ablation từng kỹ thuật |
| E5 | **Residual connection** | Mạng sâu 20–30 lớp: plain vs residual (cùng số tham số) | Training loss của hai mạng; norm gradient ở các lớp đầu |

### 4.3. Phần mở rộng — Vượt ra ngoài bài giảng

Mỗi nhóm chọn **ít nhất 2** hướng mở rộng, cài đặt, đo lường và phân tích (không chỉ "thêm vào thấy tăng"). Gợi ý:

- Chuẩn hóa: BatchNorm vs LayerNorm, vị trí pre-norm / post-norm.
- Hàm kích hoạt: GELU, SiLU, Mish, …; hàm loss: label smoothing, focal loss.
- Augmentation nâng cao: random erasing, color jitter, mixup, cutmix, AutoAugment/RandAugment.
- Kỹ thuật trọng số: EMA, Stochastic Weight Averaging (SWA), Lookahead, SAM.
- Ensemble, TTA, knowledge distillation (thầy là MLP lớn hơn do nhóm tự huấn luyện → trò MLP nhỏ).
- Tìm siêu tham số tự động (Optuna), LR range test.
- Phân tích lỗi: ma trận nhầm lẫn, các ảnh bị dự đoán sai tự tin nhất, hiệu chỉnh xác suất (calibration).
- Biến thể kiến trúc MLP: độ rộng/độ sâu, bottleneck, gated MLP (GLU), dense connection (concat). Kiến trúc chia ảnh thành patch (MLP-Mixer, …) **không** được phép vì vi phạm yêu cầu làm phẳng ảnh — nếu không chắc, hãy hỏi giảng viên trước.
