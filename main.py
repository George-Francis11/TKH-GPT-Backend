from fastapi import FastAPI
import tensorflow as tf
import requests
import uvicorn
from pydantic import BaseModel
from model import *
from fastapi.middleware.cors import CORSMiddleware
import datetime

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = GPT2(model_name="124M", run_name="TKH_Dataset_v2_May-18-2023_LearningRate_8e-05")


class TextIn(BaseModel):
    text: str


class PredictionOutput(BaseModel):
    result: str


@app.get("/")
def home():
    return {"Welcome to TKH-GPT"}


@app.get("/predict/{prefix}")
def predict(prefix: str):
    return {"result": model.predict()[0]}


@app.post("/predict")
def predict_post(payload: TextIn):
    prefix = payload.text
    result = model.predict(prefix=prefix)
    today = datetime.datetime.today().strftime('%b-%d-%Y')
    with open(f"logs/{today}.txt", "a") as f:
        f.write(f"prefix: {model.prefix}\nresponse: {model.prediction[0]}<\\endofanswer>\n")
    return {"result": model.prediction[0]}


@app.post("/test")
def test_post(payload: TextIn):
    print(payload)
    return {
        "result": "Hello world"
    }
