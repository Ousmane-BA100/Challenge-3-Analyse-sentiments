from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import os
from dotenv import load_dotenv
import logging
import time

# Charger les variables d'environnement
load_dotenv()

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="API d'Analyse de Sentiments",
             description="API pour l'analyse de sentiments en français utilisant CamemBERT",
             version="1.0.0")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En développement uniquement
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware pour logger les requêtes
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"Requête reçue: {request.method} {request.url}")
    
    try:
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        logger.info(f"Réponse: {response.status_code} (en {process_time:.2f}ms)")
        return response
    except Exception as e:
        logger.error(f"Erreur lors du traitement de la requête: {str(e)}")
        raise

# Configuration des chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Configuration du modèle
MODEL_PATH = "best_model.pt"
MODEL_NAME = 'camembert-base'
MAX_LENGTH = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Utilisation du périphérique: {DEVICE}")

# Charger le tokenizer et le modèle
try:
    logger.info("Chargement du tokenizer...")
    tokenizer = CamembertTokenizer.from_pretrained(MODEL_NAME)
    
    logger.info("Chargement du modèle...")
    model = CamembertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    logger.info("Chargement des poids du modèle...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    logger.info("Modèle chargé avec succès")
    
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
    raise

# Modèles Pydantic
class TextInput(BaseModel):
    text: str

class BatchInput(BaseModel):
    texts: List[str]

class PredictionOutput(BaseModel):
    text: str
    sentiment: str
    confidence: float

# Fonction de prédiction
def predict_sentiment(text: str) -> dict:
    try:
        logger.debug(f"Prédiction pour le texte: {text[:50]}...")
        
        # Tokenization
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Déplacer les tenseurs sur le bon périphérique
        input_ids = encoding['input_ids'].to(DEVICE)
        attention_mask = encoding['attention_mask'].to(DEVICE)
        
        # Prédiction
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Traitement des résultats
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        confidence, prediction = torch.max(probs, dim=1)
        
        result = {
            "sentiment": "POSITIVE" if prediction.item() == 1 else "NEGATIVE",
            "confidence": confidence.item()
        }
        
        logger.debug(f"Résultat de la prédiction: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Erreur dans predict_sentiment: {str(e)}")
        raise

# Configuration des templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Routes
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    logger.info("Page d'accueil demandée")
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/predict", response_model=PredictionOutput)
async def predict_single_text(text_input: TextInput):
    """Prédit le sentiment d'un seul texte"""
    try:
        logger.info(f"Début de la prédiction pour le texte: {text_input.text[:50]}...")
        start_time = time.time()
        
        result = predict_sentiment(text_input.text)
        
        response = {
            "text": text_input.text,
            "sentiment": result["sentiment"],
            "confidence": result["confidence"]
        }
        
        process_time = (time.time() - start_time) * 1000
        logger.info(f"Prédiction terminée en {process_time:.2f}ms: {response['sentiment']} ({response['confidence']:.2f})")
        
        return response
        
    except Exception as e:
        logger.error(f"Erreur dans /api/predict: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Erreur lors de l'analyse du texte",
                "message": str(e)
            }
        )

@app.post("/api/predict_batch", response_model=List[PredictionOutput])
async def predict_batch_texts(batch_input: BatchInput):
    """Prédit le sentiment de plusieurs textes en une seule requête"""
    try:
        logger.info(f"Début de la prédiction par lot pour {len(batch_input.texts)} textes")
        start_time = time.time()
        
        results = []
        for i, text in enumerate(batch_input.texts, 1):
            try:
                result = predict_sentiment(text)
                results.append({
                    "text": text,
                    "sentiment": result["sentiment"],
                    "confidence": result["confidence"]
                })
                logger.debug(f"Texte {i}/{len(batch_input.texts)} traité")
            except Exception as e:
                logger.error(f"Erreur sur le texte {i}: {str(e)}")
                raise
        
        process_time = (time.time() - start_time) * 1000
        logger.info(f"Prédiction par lot terminée en {process_time:.2f}ms")
        
        return results
        
    except Exception as e:
        logger.error(f"Erreur dans /api/predict_batch: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Erreur lors de l'analyse des textes",
                "message": str(e)
            }
        )

if __name__ == "__main__":
    import uvicorn
    logger.info("Démarrage du serveur Uvicorn...")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info",
    )