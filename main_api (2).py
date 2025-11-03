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
    if spam_model is None or toxicity_model is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    try:
        text = request.text
        
        # Spam prediction
        spam_prob = float(spam_model.predict_proba([[1]])[0][1])
        
        # Toxicity prediction
        with torch.no_grad():
            inputs = toxicity_tokenizer(text, truncation=True, max_length=128, padding='max_length', return_tensors='pt')
            outputs = toxicity_model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            pred_class = torch.argmax(logits, dim=-1).item()
            tox_score = float(probs[0][pred_class].item())
        
        labels = ['safe', 'spam', 'toxic', 'misinformation', 'unsafe']
        tox_label = labels[pred_class] if pred_class < len(labels) else 'unknown'
        
        return {
            "text": text,
            "spam_score": round(spam_prob, 4),
            "toxicity_label": tox_label,
            "toxicity_score": round(tox_score, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy" if (spam_model and toxicity_model) else "unhealthy"}

@app.get("/")
async def root():
    return {"message": "Spam & Toxicity Detection API v2.0", "endpoints": {"POST /predict": "Make prediction", "GET /docs": "API documentation"}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
