from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List
import pickle
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn
import os

app = FastAPI(
    title="Spam & Toxicity Detection API",
    description="Combined API for spam detection (XGBoost) and toxicity detection (DistilBERT)",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading Models...")

# Load Spam Detection Model
try:
    with open('spam_detection_model.pkl', 'rb') as f:
        spam_artifacts = pickle.load(f)
    
    spam_model = spam_artifacts['model']
    tfidf_vectorizer = spam_artifacts['tfidf_vectorizer']
    spam_feature_names = spam_artifacts['feature_names']
    spam_keywords = spam_artifacts['spam_keywords']
    keyword_threshold = spam_artifacts['keyword_threshold']
    
    print("Spam model loaded")
except Exception as e:
    spam_model = None

# Load Toxicity Detection Model
try:
    toxicity_model = AutoModelForSequenceClassification.from_pretrained('./toxicity_model_final')
    toxicity_tokenizer = AutoTokenizer.from_pretrained('./toxicity_model_final')
    toxicity_model.to(device)
    toxicity_model.eval()
    
    with open('toxicity_model_artifacts.pkl', 'rb') as f:
        toxicity_artifacts = pickle.load(f)
    
    label_mapping = toxicity_artifacts['label_mapping']
    reverse_mapping = toxicity_artifacts['reverse_mapping']
    
    print("Toxicity model loaded")
except Exception as e:
    toxicity_model = None

class CombinedPredictionRequest(BaseModel):
    text: str = Field(..., description="Text to analyze for spam and toxicity", min_length=1)

class SpamResponse(BaseModel):
    spam_score: float
    label_probs: dict
    found_spam_keywords: list
    keyword_analysis: dict
    explain: list

class ToxicityResponse(BaseModel):
    label: str
    toxicity_score: float

class CombinedPredictionResponse(BaseModel):
    text: str
    spam_detection: SpamResponse
    toxicity_detection: ToxicityResponse

def extract_spam_features(text: str):
    features = {}
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / (len(text) + 1)
    features['word_count'] = len(text.split())
    features['char_count'] = len(text)
    
    found_keywords = [word for word in spam_keywords if word.lower() in text.lower()]
    features['spam_keyword_count'] = len(found_keywords)
    features['has_url'] = 1 if ('http' in text.lower() or 'www' in text.lower()) else 0
    features['has_currency'] = 1 if any(char in text for char in ['$', '€', '£', '₹']) else 0
    features['has_numbers'] = 1 if any(char.isdigit() for char in text) else 0
    
    return features, found_keywords

def generate_spam_explanations(text, features, spam_prob, keyword_count, found_keywords):
    explanations = []
    
    if keyword_count >= keyword_threshold:
        keywords_str = ', '.join([f"'{kw}'" for kw in found_keywords])
        explanations.append(f"KEYWORD RULE TRIGGERED: {keyword_count} spam keywords ({keywords_str})")
    elif keyword_count > 0:
        keywords_str = ', '.join([f"'{kw}'" for kw in found_keywords])
        explanations.append(f"Contains {keyword_count} spam keyword(s): {keywords_str}")
    
    if features['exclamation_count'] >= 3:
        explanations.append(f"Too many exclamation marks ({int(features['exclamation_count'])})")
    
    if features['uppercase_ratio'] > 0.3:
        explanations.append(f"High uppercase ({features['uppercase_ratio']:.1%})")
    
    if features['has_url']:
        explanations.append("Contains URL")
    
    if not explanations:
        explanations.append("Message appears legitimate")
    
    return explanations

def predict_toxicity(text):
    with torch.no_grad():
        inputs = toxicity_tokenizer(
            text, truncation=True, max_length=128, padding='max_length', return_tensors='pt'
        )
        inputs = {key: val.to(device) for key, val in inputs.items()}
        outputs = toxicity_model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        pred_class = torch.argmax(logits, dim=-1).item()
    
    predicted_label = reverse_mapping[pred_class]
    toxicity_score = float(probabilities[0][pred_class].item())
    
    return predicted_label, toxicity_score

@app.post("/predict", response_model=CombinedPredictionResponse)
async def predict_spam_and_toxicity(request: CombinedPredictionRequest):
    
    if spam_model is None or toxicity_model is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    try:
        text = request.text
        
        text_features, found_keywords = extract_spam_features(text)
        keyword_count = text_features['spam_keyword_count']
        
        tfidf_features = tfidf_vectorizer.transform([text]).toarray()
        tfidf_dict = {f'tfidf_{i}': tfidf_features[0][i] for i in range(tfidf_features.shape[1])}
        
        feature_dict = {
            'exclamation_count': text_features['exclamation_count'],
            'question_count': text_features['question_count'],
            'uppercase_ratio': text_features['uppercase_ratio'],
            'word_count': text_features['word_count'],
            'char_count': text_features['char_count'],
            'spam_keyword_count': text_features['spam_keyword_count'],
            'has_url': text_features['has_url'],
            'has_currency': text_features['has_currency'],
            'has_numbers': text_features['has_numbers'],
            **tfidf_dict
        }
        
        features_df = pd.DataFrame([feature_dict])
        features_df = features_df[spam_feature_names]
        
        prediction_proba = spam_model.predict_proba(features_df)[0]
        not_spam_prob = float(prediction_proba[0])
        spam_prob = float(prediction_proba[1])
        
        if keyword_count >= keyword_threshold:
            spam_prob = max(spam_prob, 0.95)
            not_spam_prob = 1 - spam_prob
        
        explanations = generate_spam_explanations(
            text, text_features, spam_prob, keyword_count, found_keywords
        )
        
        spam_response = SpamResponse(
            spam_score=round(spam_prob, 4),
            label_probs={
                "spam": round(spam_prob, 4),
                "not_spam": round(not_spam_prob, 4)
            },
            found_spam_keywords=found_keywords,
            keyword_analysis={
                "keyword_count": keyword_count,
                "threshold": keyword_threshold,
                "found_keywords": found_keywords,
                "rule_triggered": keyword_count >= keyword_threshold
            },
            explain=explanations
        )
        
        toxicity_label, toxicity_score = predict_toxicity(text)
        
        toxicity_response = ToxicityResponse(
            label=toxicity_label,
            toxicity_score=round(toxicity_score, 4)
        )
        
        return CombinedPredictionResponse(
            text=text,
            spam_detection=spam_response,
            toxicity_detection=toxicity_response
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if (spam_model is not None and toxicity_model is not None) else "unhealthy",
        "spam_model_loaded": spam_model is not None,
        "toxicity_model_loaded": toxicity_model is not None,
        "device": str(device)
    }

@app.get("/")
async def root():
    return {
        "message": "Spam & Toxicity Detection API v2.0",
        "endpoints": {
            "POST /predict": "Combined prediction",
            "GET /docs": "Swagger documentation"
        }
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
