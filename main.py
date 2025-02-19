from fastapi import FastAPI
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware
import pickle

# Khởi tạo FastAPI
app = FastAPI()

# Thêm CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load dữ liệu
df_model = pd.read_pickle("dataset_encoded.pkl")
df = pd.read_csv("data_process.csv")

# Load mô hình
model = load_model("recommendation_model_fixed.h5")

# Load LabelEncoders đã lưu trước đó
with open("user_encoder.pkl", "rb") as f:
    user_encoder = pickle.load(f)
with open("track_encoder.pkl", "rb") as f:
    track_encoder = pickle.load(f)

# Kiểm tra nếu 'track_encoded' không có trong df_model
if "track_encoded" not in df_model.columns:
    raise ValueError("⚠️ Lỗi: 'track_encoded' không tồn tại trong dataset_encoded.pkl!")

@app.get("/users")
def get_users():
    return {"users": df["user_id"].unique().tolist()}

@app.get("/recommend/{user_id}")
def recommend(user_id: str):
    # Kiểm tra user_id có tồn tại không
    if user_id not in df["user_id"].values:
        return {"error": "User ID không hợp lệ!"}

    # Mã hóa user_id
    try:
        user_encoded = user_encoder.transform([user_id])[0]
    except ValueError:
        return {"error": "User ID không hợp lệ!"}

    # Lấy danh sách tất cả bài hát đã được mã hóa
    all_tracks = df_model["track_encoded"].unique()

    # Lấy danh sách bài hát user đã nghe từ df_model
    user_heard_tracks = df_model[df_model["user_id_encoded"] == user_encoded]["track_encoded"].values

    # Lọc ra các bài hát mà user chưa nghe
    unheard_tracks = np.setdiff1d(all_tracks, user_heard_tracks)

    # Nếu user đã nghe tất cả bài hát, thì chọn tất cả bài hát để dự đoán
    if len(unheard_tracks) == 0:
        unheard_tracks = all_tracks

    # Dự đoán điểm số
    user_inputs = np.full_like(unheard_tracks, user_encoded)
    predictions = model.predict([user_inputs, unheard_tracks])

    # Kiểm tra predictions có hợp lệ không
    if predictions.shape[0] == 0:
        return {"error": "Không có dữ liệu gợi ý cho user này!"}

    # Chọn top 20 bài có điểm cao nhất rồi trộn ngẫu nhiên để lấy 10 bài
    top_indices = np.argsort(predictions.flatten())[-20:]
    np.random.shuffle(top_indices)
    top_indices = top_indices[:10]


    # Trộn ngẫu nhiên để tạo đa dạng
    np.random.shuffle(top_indices)

    recommended_tracks = track_encoder.inverse_transform(unheard_tracks[top_indices])

    return {"user_id": user_id, "recommendations": recommended_tracks.tolist()}
