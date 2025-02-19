# `Collaborative Filtering - CF`

## 1. Tiền xử lý dữ liệu cho hệ thống gợi ý nhạc bằng PySpark

- Chuyển đổi `user_id` và `track` thành dạng số.
- Tính toán `rating`:
    - Nhóm theo cặp `user` và `track`.
    - Đếm số lần bài hát xuất hiện trong danh sách phát của user.
- Lưu dữ liệu đã xử lý ra file Parquet:
    - File `*.parquet` sẽ chứa dữ liệu đã tiền xử lý, sẵn sàng để huấn luyện mô hình.

### 2. Huấn luyện mô hình ALS - Alternating Least Squares

- Huấn luyện mô hình ALS để gợi ý bài hát.
- Dữ liệu đầu vào file `.parquet` chứa user_id,track và rating.
- Kết quả: Lưu mô hình đã huấn luyện vào thư mục als_model.
- Load mô hình trong API FastAPI để gợi ý bài hát cho người dùng.

# `Neutral Collaborative Filtering - NCF`

## 1. Tiền xử lý dữ liệu

- Mã hóa dữ liệu `user_id` và `track` thành số nguyên (để phù hợp với tensorflow).
- Tính số lần nghe của bài hát và chuẩn hóa bằng MinMaxScaler.
- Chuyển đổi thành TensorFlow Dataset.
- Dùng Embedding + Neutral Network để học pattern.
- Huấn luyện mô hình và kiểm tra trên tập test.

## 🔥 **So sánh Collaborative Filtering (CF) và Neural Collaborative Filtering (NCF)** 🔥  

Cả **Collaborative Filtering (CF)** và **Neural Collaborative Filtering (NCF)** đều là phương pháp phổ biến trong hệ thống gợi ý. Tuy nhiên, chúng có sự khác biệt quan trọng về cách hoạt động, hiệu suất và khả năng mở rộng.

---

## **1️⃣ Tổng quan về CF & NCF**
| **Tiêu chí**           | **Collaborative Filtering (CF)** | **Neural Collaborative Filtering (NCF)** |
|------------------------|---------------------------------|---------------------------------|
| **Cách hoạt động** | Dựa trên ma trận tương tác giữa user và item. | Dùng mạng neural network để học quan hệ user-item. |
| **Loại mô hình** | Matrix Factorization (ALS, SVD,...) | Deep Learning (MLP, Embedding layers,...) |
| **Đặc trưng đầu vào** | User-Item Matrix | Embedding user và item, thêm metadata nếu cần. |
| **Học đặc trưng (Feature Learning)** | Dựa trên phân rã ma trận (factorization). | Học đặc trưng từ dữ liệu bằng neural network. |
| **Khả năng mở rộng** | Tốt hơn với dữ liệu nhỏ và vừa. | Tốt hơn với dữ liệu lớn và phức tạp. |
| **Xử lý dữ liệu thưa (Sparse Data)** | Gặp khó khăn khi dữ liệu thưa. | Tốt hơn vì có thể học representation tốt hơn. |
| **Thêm metadata (contextual features)** | Hạn chế, chủ yếu dựa vào user-item. | Có thể dễ dàng mở rộng với thông tin bổ sung (user age, genre,...) |
| **Hiệu suất** | Nhanh hơn, ít tài nguyên hơn. | Chậm hơn, yêu cầu GPU để train tốt. |
| **Khả năng mở rộng** | Dễ triển khai, phù hợp với dữ liệu nhỏ. | Cần nhiều tài nguyên, phù hợp với dữ liệu lớn. |

---

## **2️⃣ Collaborative Filtering (CF)**
### 🔹 **Hoạt động**
- Dựa trên **Matrix Factorization (MF)** như **ALS (Alternating Least Squares)** hoặc **SVD (Singular Value Decomposition)** để phân rã ma trận tương tác user-item thành các latent factors.
- Mô hình có dạng:
  \[
  \hat{R} = U \times V^T
  \]
  Trong đó:
  - \( U \) là ma trận embedding của người dùng.
  - \( V \) là ma trận embedding của item.

### 🔹 **Ưu điểm**
✅ Hiệu quả với dữ liệu nhỏ.  
✅ Ít yêu cầu tài nguyên (không cần GPU).  
✅ Dễ triển khai với thư viện như **Spark MLlib (ALS)**.  

### 🔹 **Hạn chế**
❌ Khó xử lý **cold start problem** (user/item mới không có dữ liệu).  
❌ Không tận dụng được thông tin bổ sung (metadata, nội dung bài hát,...).  
❌ Không linh hoạt khi dữ liệu quá thưa thớt.  

### 🔹 **Khi nào dùng CF?**
🔹 Khi bạn có một hệ thống gợi ý đơn giản, dữ liệu không quá lớn.  
🔹 Khi muốn mô hình nhẹ, không cần GPU.  

---

## **3️⃣ Neural Collaborative Filtering (NCF)**
### 🔹 **Hoạt động**
- Sử dụng **Deep Learning** để học các latent representation của user và item.
- Thay vì phân rã ma trận, NCF học đặc trưng từ dữ liệu bằng cách:
  - Biểu diễn **user** và **item** dưới dạng embedding.
  - Dùng **Neural Network (MLP, CNN,...)** để tìm mối quan hệ user-item.
  - Có thể kết hợp với **context features** như độ tuổi, thể loại nhạc, thời gian nghe,...

### 🔹 **Ưu điểm**
✅ **Tốt hơn khi dữ liệu thưa thớt**, có thể tự học đặc trưng.  
✅ **Giải quyết được cold start problem** nếu có metadata (ví dụ: thể loại nhạc, quốc gia, độ tuổi người dùng,...).  
✅ **Mở rộng linh hoạt** với các thông tin bổ sung.  

### 🔹 **Hạn chế**
❌ Cần **GPU** để train nhanh.  
❌ Tốn tài nguyên hơn so với CF.  
❌ Cần nhiều dữ liệu để train tốt.  

### 🔹 **Khi nào dùng NCF?**
🔹 Khi hệ thống có **dữ liệu lớn** (hàng triệu user và item).  
🔹 Khi muốn sử dụng **thông tin bổ sung** như thể loại nhạc, thiết bị, hành vi người dùng.  
🔹 Khi có GPU và cần mô hình mạnh hơn CF truyền thống.  

---

## **4️⃣ Ví dụ minh họa**
### **🚀 Collaborative Filtering (CF) bằng ALS**
```python
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("MusicRecommendation").getOrCreate()
ratings_df = spark.read.parquet("processed_data.parquet")

als = ALS(userCol="user_index", itemCol="track_index", ratingCol="rating", nonnegative=True)
model = als.fit(ratings_df)
model.write().overwrite().save("als_model")
```
✅ **Dùng ALS để gợi ý nhạc**.  
✅ **Nhanh chóng, đơn giản**.  
✅ **Không cần GPU**.  

---

### **🚀 Neural Collaborative Filtering (NCF) bằng TensorFlow**
```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Concatenate, Dense
from tensorflow.keras.models import Model

embedding_size = 50
num_users = df_model["user_id_encoded"].nunique()
num_tracks = df_model["track_encoded"].nunique()

user_input = Input(shape=(1,), name="user_id")
track_input = Input(shape=(1,), name="track_id")

user_embedding = Embedding(input_dim=num_users, output_dim=embedding_size, name="user_embedding")(user_input)
track_embedding = Embedding(input_dim=num_tracks, output_dim=embedding_size, name="track_embedding")(track_input)

user_embedding = Flatten()(user_embedding)
track_embedding = Flatten()(track_embedding)

concat = Concatenate()([user_embedding, track_embedding])
dense1 = Dense(128, activation="relu")(concat)
dense2 = Dense(64, activation="relu")(dense1)
output = Dense(1, activation="sigmoid")(dense2)

model = Model(inputs=[user_input, track_input], outputs=output)
model.compile(loss="mse", optimizer="adam", metrics=["mae"])
model.fit(train_tensor, epochs=10, verbose=1)
```
✅ **Dùng mạng Neural Network để gợi ý nhạc**.  
✅ **Có thể thêm dữ liệu bổ sung (context-aware recommendations)**.  
✅ **Cần GPU để train nhanh**.  

---

## **5️⃣ Kết luận: Chọn CF hay NCF?**
🔹 **Dùng CF nếu**:
- Dữ liệu nhỏ hoặc vừa.
- Cần một mô hình nhẹ, dễ triển khai.
- Không cần thêm metadata như thể loại nhạc, tuổi người nghe.

🔹 **Dùng NCF nếu**:
- Dữ liệu lớn, có GPU để train.
- Muốn tận dụng thông tin bổ sung như thể loại nhạc, quốc gia, thiết bị.
- Cần mô hình mạnh hơn, linh hoạt hơn.

📌 **👉 Nếu chỉ có user_id và track_id** → Dùng CF (ALS).  
📌 **👉 Nếu có thêm thông tin về user và bài hát** → Dùng NCF.  

🚀 **NCF mạnh hơn, nhưng CF dễ triển khai hơn!**