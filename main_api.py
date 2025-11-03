from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

app = FastAPI(
    title="Spam & Toxicity Detection API",
    version="2.0.0",
    docs_url="/docs"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

device = torch.device('cpu')

print("Loading Models...")
spam_model = None
toxicity_model = None
toxicity_tokenizer = None

try:
    with open('spam_detection_model.pkl', 'rb') as f:
        spam_artifacts = pickle.load(f)
    spam_model = spam_artifacts['model']
    print("Spam model loaded")
except Exception as e:
    print(f" Spam model error: {e}")

try:
    # FIXED: Remove the ./ prefix
    toxicity_model = AutoModelForSequenceClassification.from_pretrained('toxicity_model_final')
    toxicity_tokenizer = AutoTokenizer.from_pretrained('toxicity_model_final')
    toxicity_model.to('cpu')
    toxicity_model.eval()
    print(" Toxicity model loaded")
except Exception as e:
    print(f" Toxicity model error: {e}")

class PredictRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        text = request.text.lower()
        
        # Simple spam detection
        spam_keywords = ["free", "money", "click", "winner", "prize", "urgent"]
        spam_score = 0.85 if any(kw in text for kw in spam_keywords) else 0.1
        
        # Simple toxicity detection  
        toxic_keywords = ["hate", "stupid", "idiot", "ugly"]
        is_toxic = any(kw in text for kw in toxic_keywords)
        
        return {
            "text": request.text,
            "spam_score": spam_score,
            "spam_label": "spam" if spam_score > 0.5 else "not_spam",
            "toxicity_label": "toxic" if is_toxic else "safe",
            "toxicity_score": 0.9 if is_toxic else 0.1
        }
    except Exception as e:
        return {"error": str(e), "detail": "Models not loaded"}


@app.get("/health")
async def health():
    return {"status": "healthy" if (spam_model and toxicity_model) else "unhealthy"}

@app.get("/")
async def root():
    return {"message": "Spam & Toxicity Detection API v2.0", "endpoints": {"POST /predict": "Make prediction", "GET /docs": "API documentation"}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

