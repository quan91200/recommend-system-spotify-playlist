import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data_process.csv")

# Tạo encoders
user_encoder = LabelEncoder()
track_encoder = LabelEncoder()

df["user_id_encoded"] = user_encoder.fit_transform(df["user_id"])
df["track_encoded"] = track_encoder.fit_transform(df["track"])

# Lưu encoders lại để sử dụng sau này
with open("user_encoder.pkl", "wb") as f:
    pickle.dump(user_encoder, f)

with open("track_encoder.pkl", "wb") as f:
    pickle.dump(track_encoder, f)

print("✅ Encoders đã được lưu!")
