# Expose BERT sentiment model through an API

from pathlib import Path

import torch

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "bert_model"

app = FastAPI(
    title="Airline Sentiment API",
    description="Predicts whether an airline-related message is negative, neutral, or positive.",
    version="1.0.0",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, examples=["The flight was delayed again."])


class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float


@app.get("/")
def root():
    return {"message": "Airline Sentiment API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)

    predicted_id = int(torch.argmax(probabilities, dim=-1).item())
    confidence = float(probabilities[0, predicted_id].item())

    sentiment = model.config.id2label[predicted_id]

    return PredictionResponse(
        sentiment=sentiment,
        confidence=round(confidence, 4),
    )

