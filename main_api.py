from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI(
    title="Spam & Toxicity Detection API",
    version="2.0.0"
    
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

device = torch.device("cpu")

# Load models
print("Loading models...")
spam_model, toxicity_model, toxicity_tokenizer = None, None, None

try:
    with open('spam_detection_model.pkl', 'rb') as f:
        spam_artifacts = pickle.load(f)
    spam_model = spam_artifacts['model']
    print("✅ Spam model loaded")
except Exception as e:
    print(f"❌ Spam model error: {e}")

try:
    toxicity_model = AutoModelForSequenceClassification.from_pretrained('toxicity_model_final')
    toxicity_tokenizer = AutoTokenizer.from_pretrained('toxicity_model_final')
    toxicity_model.to(device)
    toxicity_model.eval()
    print("✅ Toxicity model loaded")
except Exception as e:
    print(f"❌ Toxicity model error: {e}")

class PredictRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict(request: PredictRequest):
    if not spam_model or not toxicity_model or not toxicity_tokenizer:
        raise HTTPException(status_code=500, detail="Models not loaded")
    try:
        text = request.text
        # --- Spam keyword rule-based detection ---
        spam_keywords = ["free", "money", "win", "winner", "urgent", "prize", "offer", "click", "congratulations"]
        found_keywords = [kw for kw in spam_keywords if kw in text.lower()]
        keyword_count = len(found_keywords)
        rule_triggered = keyword_count > 0
        
        # Model-driven probabilities
        spam_label_probs = dict(spam=0, not_spam=1)
        spam_score = 0
        if spam_model:
            # Here: update this part to match your spam model's requirements
            pred_prob = spam_model.predict_proba([[1]])[0]
            spam_label_probs = {
                "spam": float(pred_prob[1]),
                "not_spam": float(pred_prob[0])
            }
            spam_score = float(pred_prob[1])

        spam_explain = []
        if rule_triggered:
            spam_explain.append(f"🚨 KEYWORD RULE: {keyword_count} spam keyword{'s' if keyword_count>1 else ''}")

        # --- Toxicity prediction ---
        labels = ['safe', 'spam', 'toxic', 'misinformation', 'unsafe']
        tox_label = 'unknown'
        tox_score = 0.0
        all_scores = {l: 0 for l in labels}
        confidence = 0.0

        if toxicity_model and toxicity_tokenizer:
            with torch.no_grad():
                inputs = toxicity_tokenizer(text, truncation=True, max_length=128, padding='max_length', return_tensors='pt')
                outputs = toxicity_model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                pred_class = torch.argmax(logits, dim=-1).item()
                tox_label = labels[pred_class] if pred_class < len(labels) else "unknown"
                tox_score = float(probs[0][pred_class])
                confidence = tox_score
                all_scores = {labels[i]: float(probs[0][i]) for i in range(len(labels))}

        return {
            "spam_detection": {
                "label_probs": spam_label_probs,
                "explain": spam_explain,
                "keyword_analysis": {
                    "keyword_count": keyword_count,
                    "found_keywords": found_keywords,
                    "rule_triggered": rule_triggered
                }
            },
            "toxicity_detection": {
                "label": tox_label,
                "toxicity_score": tox_score,
                "confidence": confidence,
                "all_scores": all_scores
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy" if (spam_model and toxicity_model and toxicity_tokenizer) else "unhealthy"}

@app.get("/")
async def root():
    return {
        "message": "Spam & Toxicity Detection API v2.0",
        "endpoints": {
            "POST /predict": "Make prediction",
            "GET /docs": "API documentation"
        }
    }
