# HƯỚNG DẪN CHẠY DỰ ÁN "MLP CHALLENGE" (CIFAR-10)

Tài liệu này hướng dẫn chi tiết cách chạy và sử dụng các file trong dự án để đạt kết quả tối đa theo đúng yêu cầu của đề bài [`DE_BAI.md`](file:///d:/3/DuDoanHinhAnh2/DE_BAI.md).

---

## 📌 1. Danh sách các File chính trong Dự án

| Tên File | Mục đích & Mô tả |
| :--- | :--- |
| **[`MLP_Challenge_Experiments.ipynb`](file:///d:/3/DuDoanHinhAnh2/MLP_Challenge_Experiments.ipynb)** | **(KHUYÊN DÙNG ĐỂ LÀM BÁO CÁO)** Chứa toàn bộ 5 thực nghiệm có kiểm soát bắt buộc (**E1 $\rightarrow$ E5**), 2 hướng mở rộng, vẽ sẵn toàn bộ biểu đồ khoa học và xuất file checkpoint `.pt`. |
| **[`Baitap2dudoanhinhanh_MLP.ipynb`](file:///d:/3/DuDoanHinhAnh2/Baitap2dudoanhinhanh_MLP.ipynb)** | Pipeline huấn luyện Stratified 3-Fold Ensemble thuần MLP để tối ưu điểm Accuracy nộp bài (`submission_mlp.csv`). |
| **[`demo_app_mlp.py`](file:///d:/3/DuDoanHinhAnh2/demo_app_mlp.py)** | Ứng dụng Web Demo (Gradio) nhận diện ảnh qua Webcam hoặc Upload, kết nối trực tiếp với mô hình MLP đã train. |
| **[`Baitap2dudoanhinhanh-2.ipynb`](file:///d:/3/DuDoanHinhAnh2/Baitap2dudoanhinhanh-2.ipynb)** | File gốc cũ của nhóm (kiến trúc CNN ResNet — được giữ nguyên vẹn để đối chiếu). |
| **[`baseline.py`](file:///d:/3/DuDoanHinhAnh2/baseline.py)** | File baseline mẫu của giảng viên (MLP 1 lớp ẩn Sigmoid). |

---

## ⚙️ 2. Cấu hình Môi trường thực thi trong VS Code

Môi trường Python ảo đã được tích hợp đầy đủ PyTorch, CUDA Toolkit (RTX 4060), Matplotlib và Gradio:

* **Đường dẫn Python Interpreter / Kernel:**
  ```text
  D:\4-University Stuff Season 16\KI 1\Mang Noron\DuDoanHinhAnh\ai_env\Scripts\python.exe
  ```

### Các bước chọn Kernel:
1. Mở file notebook (`.ipynb`) trong VS Code.
2. Nhìn góc trên bên phải, nhấn vào **Select Kernel** $\rightarrow$ **Python Environments...**.
3. Chọn môi trường `ai_env` (nếu không thấy, chọn *Enter interpreter path* và dán đường dẫn ở trên vào).

---

## 🚀 3. Hướng dẫn chạy từng File

### 🅰️ Chạy Notebook Thí nghiệm Khoa học: `MLP_Challenge_Experiments.ipynb`
> **Dành cho:** Lấy số liệu, vẽ đồ thị để viết Báo cáo Word / Slide thuyết trình (chuẩn 100% đề bài).

1. Mở file [`MLP_Challenge_Experiments.ipynb`](file:///d:/3/DuDoanHinhAnh2/MLP_Challenge_Experiments.ipynb).
2. Nhấn **Run All** (hoặc chạy lần lượt từng cell bằng `Shift + Enter`).
3. **Các kết quả tự động hiển thị:**
   * **Mục 1 (Baseline):** Tái lập mô hình 1 lớp của thầy (Acc ~38%).
   * **Mục 2 (E1 - Optimizer):** Đồ thị so sánh 5 thuật toán (`SGD`, `Momentum`, `RMSprop`, `Adam`, `AdamW`).
   * **Mục 3 (E2 - Khởi tạo trọng số):** Biểu đồ đo độ lệch chuẩn activation qua 10 lớp sâu (chứng minh tiêu biến/bùng nổ gradient).
   * **Mục 4 (E3 - Lịch LR):** Đồ thị so sánh Fixed vs Step Decay vs Cosine vs Warmup+Cosine.
   * **Mục 5 (E4 - Regularization):** Bảng Ablation Study thu hẹp khoảng cách Overfitting.
   * **Mục 6 (E5 - Residual Connection):** Đồ thị chứng minh Residual giải cứu mạng sâu 20 lớp so với Plain MLP.
   * **Mục 7 (Phần mở rộng):** So sánh BatchNorm1d vs LayerNorm.
   * **Mục 8 (Champion Model):** Huấn luyện mô hình tối ưu, lưu `best_mlp_checkpoint.pt` và xuất file nộp bài `submissions/submission_mlp_experiments.csv`.
   * **Mục 9 (Ma trận nhầm lẫn):** Vẽ Confusion Matrix trực quan 10 lớp vật thể.
4. **Thời gian chạy:** Chỉ khoảng **15 – 20 phút** trên GPU RTX 4060.

---

### 🅱️ Chạy Demo Web App: `demo_app_mlp.py`
> **Dành cho:** Trình diễn trực quan trước giảng viên và lớp học.

Mở PowerShell tại thư mục dự án và chạy lệnh:
```powershell
& "D:\4-University Stuff Season 16\KI 1\Mang Noron\DuDoanHinhAnh\ai_env\Scripts\python.exe" demo_app_mlp.py
```
* Trình duyệt sẽ mở giao diện Gradio tại địa chỉ `http://127.0.0.1:7860`.
* Bạn có thể tải ảnh bất kỳ từ máy tính hoặc bật webcam để mô hình Deep Residual MLP dự đoán top 3 nhãn trong thời gian thực.
* *(Nếu chưa chạy train xong notebook, bạn có thể thêm cờ `--dummy` để test thử giao diện: `python demo_app_mlp.py --dummy`).*

---

## 💡 4. Những câu hỏi lý thuyết quan trọng cần lưu ý khi Báo cáo

1. **Tại sao đề bài bắt buộc làm phẳng ảnh sang vector 3072 chiều mà không cho dùng CNN?**
   * *Trả lời:* Môn học muốn sinh viên hiểu sâu sắc về mạng nơ-ron nhiều lớp (MLP) thuần túy, hiểu được giới hạn khi mất thông tin cấu trúc không gian 2D, và rèn luyện kỹ năng áp dụng các kỹ thuật cốt lõi (E1 - E5) để tối ưu một kiến trúc khó.
2. **Tại sao AdamW tốt hơn Adam trong E1?**
   * *Trả lời:* Adam truyền thống áp dụng L2 regularization vào gradient của hàm mất mát khiến weight decay bị triệt tiêu bởi ma trận phương sai bậc hai. AdamW tách biệt hoàn toàn bước trừ trọng số (decoupled weight decay), giúp tối ưu hóa chuẩn xác hơn.
3. **Tại sao Residual Connection (E5) lại cứu được mạng sâu 20 lớp?**
   * *Trả lời:* Đạo hàm của hàm tắt $\frac{\partial (F(x) + x)}{\partial x} = \frac{\partial F(x)}{\partial x} + 1$. Số hạng $+1$ tạo thành "đường cao tốc" (gradient highway) cho phép gradient truyền ngược trực tiếp về các lớp đầu tiên mà không bị suy giảm theo cấp số nhân.

