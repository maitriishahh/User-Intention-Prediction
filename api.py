from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from user_intention_prediction.pipeline.prediction_pipeline import PredictionPipeline

# Initialize app
app = FastAPI(
    title="User Purchase Prediction API",
    description="Predict user purchase intention",
    version="1.0"
)

# Initialize pipeline
pipeline = PredictionPipeline()
model = pipeline.load_model()


# Request Schema
class UserInput(BaseModel):

    Administrative: float
    Informational: float
    ProductRelated: float
    BounceRates: float
    ExitRates: float
    PageValues: float
    Month: str
    VisitorType: str
    Weekend: int


@app.get("/")
def home():
    return {"message": "User Intention Prediction API Running"}


@app.post("/predict")
def predict(data: UserInput):

    user_input = data.dict()

    prediction, probability = pipeline.predict(model, user_input)

    return {
        "prediction": int(prediction[0]),
        "probability": float(probability[0])
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)