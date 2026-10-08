Tôi đã tổng hợp lại toàn bộ ngữ cảnh, tiến trình kỹ thuật và kết quả của dự án dưới dạng bản tóm tắt đầy đủ, mạch lạc để bạn dễ dàng copy sang đoạn chat mới mà không bị mất thông tin:

---

### TÓM TẮT DỰ ÁN HUẤN LUYỆN MÔ HÌNH (MLP CHALLENGE 2)

#### 1. Thông tin bài toán & Mục tiêu

* **Môn học:** Mạng nơ-ron (Neural Networks / Deep Learning) tại trường đại học.


* **Tên bài tập:** **"MLP Challenge 2": Nhận dạng ảnh thật (CIFAR-10) bằng mạng nơron nhiều lớp** (`Baitap2dudoanhinhanh.ipynb`).


* **Nhiệm vụ:** Phân loại ảnh màu $32 \times 32$ thuộc 10 lớp vật thể/con vật của bộ dữ liệu CIFAR-10 (máy bay, ô tô, chim, mèo, hươu, chó, ếch, ngựa, tàu thủy, xe tải).


* **Thang đo đánh giá:** Độ chính xác (Accuracy) trên tập kiểm tra ẩn gồm **10.000 ảnh test**.


* **Mục tiêu của người dùng:** Thiết lập môi trường chạy mô hình trực tiếp trên máy cá nhân để hỗ trợ bạn cùng lớp hoàn thành bài nộp chất lượng cao trước deadline (thứ Sáu), thay vì phụ thuộc vào giới hạn ngắt kết nối (runtime limit) của Google Colab.



---

#### 2. Cấu hình phần cứng & Môi trường thực thi

* **Thiết bị:** Laptop **Lenovo Legion Slim 5 (2023)**.
* **Cấu hình phần cứng:** CPU AMD Ryzen 7 7840HS, GPU NVIDIA GeForce RTX 4060 Laptop (8GB VRAM).


* **Môi trường phần mềm:**
* Hệ điều hành Windows, mã nguồn chạy qua VS Code với Jupyter Notebook.


* Môi trường ảo Python: `ai_env`.


* Khung làm việc: PyTorch tích hợp CUDA hoàn chỉnh.




* **Các lỗi kỹ thuật đã xử lý thành công trước đó:** Sửa lỗi biến môi trường PATH, cấu hình tương thích PyTorch với CUDA Toolkit, mã hóa đường dẫn thư mục tiếng Việt và tránh treo tiến trình đa luồng (`num_workers` deadlock).

---

#### 3. Thiết lập mô hình & Chiến lược huấn luyện

* **Chiến lược kiểm định:** K-Fold Cross Validation với $K = 3$ (`N_SPLITS = 3`).


* **Số Epoch & Dừng sớm:** Đặt trần `EPOCHS = 80`, sử dụng **Early Stopping** (`PATIENCE = 7`) dựa trên hàm mất mát tập validation để tránh lãng phí thời gian và chống quá khớp.


* **Kiểm soát Overfitting:** Tăng tỷ lệ Dropout lên `DROPOUT_RATE = 0.3` kết hợp bộ điều chỉnh tốc độ học (`lr_scheduler`).


* **Dự đoán cuối cùng:** Tính trung bình xác suất suy luận từ cả 3 mô hình (`avg_test_probs = np.mean(test_preds_folds, axis=0)`) rồi lấy `argmax` để ensemble kết quả tối ưu.



---

#### 4. Hiệu năng & Chỉ số phần cứng khi chạy thực tế

* **Tốc độ:** ~42s – 45s / epoch (nhanh hơn đáng kể so với mức ~75s / epoch của Google Colab T4).


* **Tải & Nhiệt độ:** GPU RTX 4060 hoạt động ổn định ở mức 36% – 95% công suất, nhiệt độ duy trì mát mẻ chỉ từ **54°C đến 65°C**, VRAM sử dụng ổn định ở mức **3.3 / 8.0 GB**.


* **Tổng thời gian chạy:** **177 phút 46 giây** (~2 giờ 57 phút) cho trọn vẹn cả 3 fold.


* **Điện năng tiêu thụ ước tính:** Công suất toàn máy khoảng 130W, tiêu thụ khoảng $0.26 \text{ kWh}$ điện (~780 VNĐ).

---

#### 5. Kết quả huấn luyện & Kiểm tra file nộp bài

* **Chỉ số hội tụ trên 3 Fold:**
* **Mean Val ACC:** **$0.8284 \pm 0.0028$** (~82.84% độ chính xác trung bình trên tập validation).


* **Mean Val RMSE:** **$0.1583 \pm 0.0022$**.


* Độ lệch chuẩn cực nhỏ khẳng định mô hình có tính ổn định rất cao và hội tụ đồng đều trên từng fold.




* **Kiểm định file kết quả `submissions/submission_5.csv`:**
* Kích thước: Đúng chuẩn **10.000 dòng**, gồm 2 cột `id` và `label`.


* Dữ liệu sạch: **0 giá trị khuyết thiếu (Null/NaN)**.
* Phân phối nhãn: Cân bằng lý tưởng trên cả 10 nhãn (mỗi nhãn chiếm xấp xỉ 948 – 1.064 mẫu), xác nhận mô hình không bị thiên vị lớp.