`uvicorn main:app --reload`

`npm run dev`

## **Collaborative Filtering (CF) và Neural Collaborative Filtering (NCF)**  

Trong bài toán gợi ý nhạc với **dataset gồm 4 cột**:  
- **`user_id`**: Người dùng  
- **`artist`**: Nghệ sĩ  
- **`track`**: Bài hát  
- **`playlist`**: Danh sách phát  

Áp dụng hai phương pháp chính để huấn luyện mô hình:  

---

### **1️⃣ Collaborative Filtering (CF) - Lọc cộng tác**  
Đây là phương pháp truyền thống dựa vào **hành vi của người dùng** để dự đoán sở thích.  

**👉 Có 2 loại chính:**  
- **User-Based CF**: Dự đoán dựa trên người dùng tương tự.  
- **Item-Based CF**: Dự đoán dựa trên bài hát tương tự.  

💡 **Mô hình: Alternating Least Squares (ALS) - Item-Based CF**  
- Sử dụng thuật toán **ALS** (PySpark) để huấn luyện mô hình dựa trên ma trận `user_id` × `track` với trọng số là số lần xuất hiện bài hát trong danh sách phát (**playlist**).  
- **Lợi ích**: Tính toán nhanh, dễ triển khai, không cần nhiều tài nguyên.  
- **Hạn chế**: Chỉ sử dụng thông tin từ `user_id`, `track`, có thể bị vấn đề **Cold Start** với user/bài hát mới.  

---

### **2️⃣ Neural Collaborative Filtering (NCF) - Lọc cộng tác bằng mạng nơ-ron**  
Phương pháp **NCF** sử dụng mạng nơ-ron sâu để học **biểu diễn nhúng (embedding)** của người dùng và bài hát.  

📌 **Cấu trúc NCF gồm 2 thành phần:**  
- **Generalized Matrix Factorization (GMF)**: Biến thể của CF dùng **hàm nhân ma trận (dot product)** trên embeddings.  
- **Multi-Layer Perceptron (MLP)**: Học đặc trưng phức tạp hơn từ user và track embeddings.  

💡 **Mô hình: NCF với TensorFlow**  
- **Chuẩn bị dữ liệu**:  
  - Dùng `LabelEncoder` để chuyển `user_id`, `track` thành số (`user_id_encoded`, `track_encoded`).  
  - Tạo trọng số `play_count` dựa trên số lần xuất hiện trong `playlist`.  
  - Dùng `MinMaxScaler` để chuẩn hóa `play_count`.  
- **Huấn luyện**:  
  - Dùng **Embedding Layers** cho `user_id` và `track`.  
  - Kết hợp với **MLP** để học tương tác phi tuyến giữa user và bài hát.  
  - Tối ưu bằng **Adam + MSE Loss**.  

✅ **Lợi ích của NCF:**  
- Không cần ma trận thưa (sparse matrix), giảm ảnh hưởng của **Cold Start**.  
- Học được **mối quan hệ phi tuyến** giữa người dùng và bài hát.  
- Có thể kết hợp nhiều thông tin hơn như **artist, playlist** để tăng độ chính xác.  

❌ **Hạn chế:**  
- Cần **nhiều tài nguyên** để huấn luyện, lâu hơn ALS.  
- Cần nhiều dữ liệu để đạt hiệu suất cao.  

---

### **🤔 Nên chọn CF hay NCF?**
| Tiêu chí              | Collaborative Filtering (ALS) | Neural Collaborative Filtering (NCF) |
|----------------------|--------------------------|--------------------------------|
| **Tốc độ**           | ✅ Nhanh (dùng Spark)    | ❌ Chậm (Deep Learning)       |
| **Dữ liệu đầu vào**  | `user_id`, `track`, `rating` | `user_id`, `track`, `playlist`, `artist`, `rating` |
| **Cold Start**       | ❌ Không tốt               | ✅ Tốt hơn nhờ embeddings |
| **Tài nguyên**      | ✅ Ít                     | ❌ Tốn GPU/CPU nhiều hơn |
| **Mô hình đơn giản** | ✅ Dễ triển khai         | ❌ Phức tạp hơn |