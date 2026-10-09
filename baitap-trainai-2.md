
---

# NHẬT KÝ DỰ ÁN: "MLP CHALLENGE" — CIFAR-10 BẰNG MẠNG NƠ-RON NHIỀU LỚP

### 1. Thông tin bài toán & Mục tiêu học thuật

* **Môn học:** Mạng nơ-ron và Học sâu (Neural Networks & Deep Learning).


* **Tên đề tài:** "MLP Challenge" — Nhận dạng ảnh thật (CIFAR-10) bằng mạng nơ-ron nhiều lớp.


* **Bộ dữ liệu:** CIFAR-10 gồm $50.000$ ảnh huấn luyện và $10.000$ ảnh kiểm tra ẩn nhãn, kích thước gốc $32 \times 32 \times 3$ thuộc 10 lớp vật thể/con vật.


* **Yêu cầu cốt lõi của đề bài:**
* Ảnh bắt buộc phải làm phẳng thành vector **3072 chiều** ($32 \times 32 \times 3$) trước khi đưa vào các tầng tính toán.


* Kiến trúc mô hình bắt buộc là Mạng nơ-ron nhiều lớp kết nối đầy đủ (Multi-Layer Perceptron — MLP thuần túy sử dụng các tầng `nn.Linear`).


* Nghiêm cấm hoàn toàn việc sử dụng mạng tích chập (CNN/`Conv2d`) hoặc các kiến trúc chia patch không gian như MLP-Mixer.




* **Thang đo đánh giá:** Độ chính xác phân loại (Accuracy) trên tập kiểm tra ẩn.



---

### 2. Môi trường phần cứng & Thiết lập hệ thống

* **Thiết bị thực thi:** Laptop cá nhân Lenovo Legion Slim 5 (2023).


* **Phần cứng:** CPU AMD Ryzen 7 7840HS, GPU NVIDIA GeForce RTX 4060 Laptop (8GB VRAM).


* **Môi trường phần mềm:**
* Hệ điều hành Windows, phát triển qua VS Code với Jupyter Notebook.


* Môi trường ảo Python: `ai_env` (Python 3.10.6).


* Nền tảng framework: PyTorch tích hợp CUDA hoàn chỉnh, tối ưu hóa qua Mixed Precision (AMP).




* **Sự cố kỹ thuật đã giải quyết:** Đã cấu hình chuẩn hóa biến môi trường PATH, khắc phục lỗi tương thích PyTorch CUDA Toolkit, xử lý đường dẫn thư mục tiếng Việt và tránh xung đột tiến trình đa luồng (`num_workers=0`) trên Windows.



---

### 3. Lịch sử phát triển & Bước ngoặt tái cấu trúc kiến trúc

#### Giai đoạn 1: Phiên bản CNN ban đầu (`Baitap2dudoanhinhanh-2.ipynb`)



* **Thiết kế:** Xây dựng mô hình `DeepResNetModel` dựa trên các khối tích chập 2 chiều (`nn.Conv2d`, `BatchNorm2d`, `MaxPool2d`).


* **Kết quả:** Đạt Mean Val Acc **92.84%** và Val RMSE **0.1103** sau 3 fold huấn luyện (mỗi epoch mất 42s – 45s).


* **Đánh giá vi phạm:** Tôi và nhóm đã rà soát lại đề bài và phát hiện việc dùng mạng tích chập vi phạm trực tiếp quy định của "MLP Challenge", tiềm ẩn nguy cơ bị hủy điểm toàn bộ phần mô hình.



#### Giai đoạn 2: Tái cấu trúc chuẩn mực sang Deep Residual MLP (`Baitap2dudoanhinhanh_MLP.ipynb`)



* **Khắc phục triệt để:** Loại bỏ toàn bộ tầng tích chập và pooling. Toàn bộ dữ liệu sau khi biến đổi tăng cường (Data Augmentation) đều được duỗi phẳng thành vector 3072 chiều (`torch.flatten`).


* **Xây dựng ResMLP:** Thiết kế mô hình Deep Residual MLP thuần túy với các khối kết nối đầy đủ (`nn.Linear`), chuẩn hóa 1 chiều (`nn.BatchNorm1d`), kích hoạt phi tuyến `nn.GELU` và đường tắt phần dư (residual shortcut).



---

### 4. So sánh kỹ thuật giữa 3 phiên bản mã nguồn

| Tiêu chí / Đặc tả kỹ thuật | 1. File gốc của thầy (`baseline.py`) | 2. File CNN cũ (`Baitap2dudoanhinhanh-2.ipynb`) | 3. File chuẩn mới (`Baitap2dudoanhinhanh_MLP.ipynb`) |
| --- | --- | --- | --- |
| **Định dạng file** | `.py` | `.ipynb` | `.ipynb` |
| **Tính hợp lệ với đề bài** | Hợp lệ (dạng MLP cơ bản) | Phạm quy (dùng CNN tích chập) | Hợp lệ 100% (chuẩn Deep ResMLP) |
| **Dữ liệu đầu vào** | Vector phẳng 3072 chiều | Tensor ảnh 2D $(3, 32, 32)$ | Vector phẳng 3072 chiều |
| **Độ sâu kiến trúc** | 1 tầng ẩn ($3072 \rightarrow 256 \rightarrow 10$) | 4 stage Conv2d sâu | 12 tầng Linear (1 in-proj + 5 khối ResMLP + 1 classifier) |
| **Kết nối phần dư** | Không có | Có (trên tensor 2D) | Có ($y = \text{GELU}(\text{block}(x) + x)$ trên vector 1D) |
| **Hàm kích hoạt** | `nn.Sigmoid()` | `nn.ReLU()` | `nn.GELU()` |
| **Chuẩn hóa tầng** | Không có | `nn.BatchNorm2d` | `nn.BatchNorm1d` |
| **Bộ tối ưu (Optimizer)** | SGD thuần ($lr = 0.01$) | AdamW ($lr = 10^{-3}$, $wd = 2 \cdot 10^{-2}$) | AdamW ($lr = 10^{-3}$, $wd = 2 \cdot 10^{-2}$) |
| **Lịch học (LR Scheduler)** | Cố định | Warmup 3 epoch + Cosine Annealing | Warmup 3 epoch + Cosine Annealing |
| **Chống quá khớp** | Không có | Dropout 0.3, Augmentation, Weight Decay | Dropout 0.25, Augmentation, Weight Decay, Early Stopping |
| **Kiểm định chéo** | Hold-out đơn lẻ (5.000 ảnh val) | Stratified 3-Fold Cross Validation | Stratified 3-Fold Cross Validation |
| **Dự đoán kiểm tra** | 1 mô hình đơn | Ensemble Soft-voting 3 fold | Ensemble Soft-voting 3 fold |
| **Tối ưu phần cứng** | FP32 chuẩn | FP32 chuẩn (chưa mở AMP) | Mixed Precision (AMP autocast + GradScaler) |
| **Tốc độ trên RTX 4060** | ~Vài giây/epoch | ~42s – 45s / epoch | ~31s – 32s / epoch |
| **Độ chính xác (Val Acc)** | ~35% – 40% | 92.84% (kết quả ảo do sai kiến trúc) | 59.03% $\pm$ 0.39% (chuẩn mực cao cho MLP) |
| **Tệp tin kết quả** | `submissions/baseline.csv` | `submissions/submission_5.csv` | `submissions/submission_mlp.csv` |

---

### 5. Kết quả huấn luyện chi tiết của mô hình Deep Residual MLP

* **Chi tiết quá trình hội tụ qua 3 Fold:**
* **Fold 1:** Val Acc cao nhất đạt **59.32%**, Val RMSE đạt **0.2327**.


* **Fold 2:** Val Acc cao nhất đạt **59.30%**, Val RMSE đạt **0.2320**.


* **Fold 3:** Val Acc cao nhất đạt **58.48%**, Val RMSE đạt **0.2333**.




* **Chỉ số tổng hợp (OOF):**
* **Mean Val Accuracy:** **59.03% $\pm$ 0.39%**.


* **Mean Val RMSE:** **0.2326 $\pm$ 0.0005**.




* **Kiểm định tệp xuất nộp bài (`submissions/submission_mlp.csv`):**
* Tổng số lượng mẫu: Đúng chuẩn $10.000$ dòng.


* Chất lượng dữ liệu: Đầy đủ 2 cột `id` và `label`, không có giá trị khuyết thiếu (NaN/Null).


* Phân phối nhãn dự đoán: Cân bằng lý tưởng trên cả 10 lớp, dao động từ $700$ đến $1.197$ mẫu/lớp (Lớp 0: 924, Lớp 1: 1103, Lớp 2: 700, Lớp 3: 901, Lớp 4: 909, Lớp 5: 989, Lớp 6: 1197, Lớp 7: 1141, Lớp 8: 1045, Lớp 9: 1091).





---

### 6. Nhiệm vụ trọng tâm tiếp theo để hoàn thiện bài tập lớn

Tôi xác định hai nhóm công việc còn lại cần hoàn thiện theo đề bài (`DE_BAI.md`):

* **Triển khai 5 thí nghiệm so sánh có kiểm soát (E1 – E5) để lấy số liệu và vẽ đồ thị báo cáo:**
1. *E1 (Optimizer):* So sánh đối chứng SGD, Momentum, RMSProp, Adam, AdamW.


2. *E2 (Khởi tạo trọng số):* Khảo sát mạng sâu $\ge 10$ lớp Linear (không residual, không norm) với trọng số quá nhỏ, quá lớn, Xavier, He kèm đo độ lệch chuẩn gradient.


3. *E3 (Lịch Learning Rate):* So sánh LR cố định, Step Decay, Cosine, Warmup + Cosine.


4. *E4 (Ablation Regularization):* Đo lường tác động khi loại bỏ/thêm lần lượt Weight Decay, Dropout, Data Augmentation, Early Stopping.


5. *E5 (Residual Connection):* Đo độ lớn norm gradient ở các lớp đầu giữa Plain MLP 20 lớp vs ResMLP 20 lớp để minh họa hiện tượng triệt tiêu gradient.




* **Thực hiện phần mở rộng (Chọn ít nhất 2 hướng):**
* So sánh cơ chế chuẩn hóa: `BatchNorm1d` vs `LayerNorm` trên MLP.


* Tích hợp Label Smoothing Loss hoặc kỹ thuật trộn ảnh Mixup trên vector phẳng.
