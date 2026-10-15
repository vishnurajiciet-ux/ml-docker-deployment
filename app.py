
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

app = FastAPI(title="ML Prediction API")

# Load data and train a simple model
iris = load_iris()
model = LogisticRegression(max_iter=200)
model.fit(iris.data, iris.target)

class FlowerInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def home():
    return {"message": "ML model is running in Docker!"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: FlowerInput):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = int(model.predict(features)[0])

    return {
        "predicted_class": prediction,
        "flower_name": iris.target_names[prediction].item()
    }