from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# import function from 
from mypreprocessing import preprocess_data

class Request(BaseModel):
    feature_a: int
    feature_b: int
    feature_c: int
    feature_d: int
    feature_e: int
    feature_f: float
    feature_g: float
    feature_h: int

class Response(BaseModel):
    model: str
    pred: int


filename = "models/model_diabetes.pkl"
model = pickle.load(open(filename, "rb"))

app = FastAPI()

@app.post("/predict", response_model=Response)
def predict(item: Request):

    data = [
        [
            item.feature_a, 
            item.feature_b, 
            item.feature_c, 
            item.feature_f, 
            item.feature_e, 
            item.feature_f, 
            item.feature_g, 
            item.feature_h
        ]
    ]

    x = preprocess_data(data)


    
    prediction = model.predict(x)
    
    prediction_response = {
                "model": "diabetes",
                "pred": prediction[0]
            }
    
    return prediction_response
