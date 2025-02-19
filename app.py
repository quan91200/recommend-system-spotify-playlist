from fastapi import FastAPI
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
import uvicorn

spark = SparkSession.builder.appName("MusicRecommendationAPI").getOrCreate()

app = FastAPI()

model_path = "als_model"
model = ALSModel.load(model_path)

@app.get("/")
def home():
    return {"message": "Music Recommendation API is running!"}

@app.get("/recommend/{user_id}")
def recommend_songs(user_id: str):
    user_df = spark.createDataFrame([(user_id,)], ["user_index"])

    recommendations = model.recommendForUserSubset(user_df, 10)

    rec_list = recommendations.select("recommendations.track_index").collect()
    recommended_tracks = [row.track_index for row in rec_list[0]["recommendations"]]

    return {"user_id": user_id, "recommended_tracks": recommended_tracks}

@app.get("/users")
def get_users():
    return {"users": df["user_id"].unique().tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)