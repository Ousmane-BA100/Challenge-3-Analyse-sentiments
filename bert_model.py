import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from pymongo import MongoClient
from dotenv import load_dotenv
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Charger les variables d'environnement
load_dotenv()

# Configuration
MODEL_NAME = 'camembert-base'
MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 2e-5
WARMUP_STEPS = 100

# Vérification du GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Utilisation du dispositif: {device}")

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class BERTSentimentAnalyzer:
    def __init__(self):
        # Charger le tokenizer et le modèle
        self.tokenizer = CamembertTokenizer.from_pretrained(MODEL_NAME)
        self.model = CamembertForSequenceClassification.from_pretrained(
            MODEL_NAME, 
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False
        ).to(device)
        
        # Charger les données depuis MongoDB
        self.load_data()
    
    def load_data(self):
        # Connexion à MongoDB
        client = MongoClient(os.getenv('MONGO_URI'))
        db = client[os.getenv('MONGO_DB')]
        collection = db[os.getenv('MONGO_COLLECTION')]
        
        # Récupérer les données
        data = list(collection.find({}, {'_id': 0, 'text': 1, 'label': 1}))
        df = pd.DataFrame(data)
        
        if df.empty:
            raise ValueError("Aucune donnée trouvée dans la collection MongoDB. Exécutez d'abord mongo_importer.py")
        
        # Afficher la répartition des classes
        print("\nRépartition des classes dans le jeu de données:")
        print(df['label'].value_counts())
        
        # Division des données
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df['text'].values,
            df['label'].values,
            test_size=0.2,
            random_state=42,
            stratify=df['label'].values
        )
        
        # Création des datasets
        self.train_dataset = SentimentDataset(
            train_texts, train_labels, self.tokenizer, MAX_LENGTH)
        self.val_dataset = SentimentDataset(
            val_texts, val_labels, self.tokenizer, MAX_LENGTH)
        
        # Création des DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    def train(self):
        optimizer = AdamW(self.model.parameters(), lr=LEARNING_RATE, eps=1e-8)
        
        # Nombre total d'étapes d'entraînement
        total_steps = len(self.train_loader) * EPOCHS
        
        # Création du scheduler pour le learning rate
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=WARMUP_STEPS,
            num_training_steps=total_steps
        )
        
        print(f"\nDébut de l'entraînement sur {len(self.train_loader)} batches par époque...")
        
        # Pour enregistrer le meilleur modèle
        best_accuracy = 0
        
        for epoch in range(EPOCHS):
            self.model.train()
            total_train_loss = 0
            
            # Barre de progression
            progress_bar = tqdm(self.train_loader, desc=f'Époque {epoch + 1}/{EPOCHS}')
            
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                self.model.zero_grad()
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_train_loss += loss.item()
                
                loss.backward()
                
                # Éviter l'explosion du gradient
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                
                progress_bar.set_postfix({'loss': loss.item()})
            
            # Calcul de la perte moyenne sur l'ensemble d'entraînement
            avg_train_loss = total_train_loss / len(self.train_loader)
            
            # Évaluation sur l'ensemble de validation
            val_loss, val_accuracy, val_report = self.evaluate()
            
            print(f"\nÉpoque {epoch + 1}:")
            print(f"  Perte d'entraînement: {avg_train_loss:.4f}")
            print(f"  Perte de validation: {val_loss:.4f}")
            print(f"  Précision de validation: {val_accuracy:.4f}")
            print("\nRapport de classification sur la validation:")
            print(val_report)
            
            # Sauvegarder le meilleur modèle
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(self.model.state_dict(), 'best_model.pt')
                print(f"Nouveau meilleur modèle sauvegardé avec une précision de {best_accuracy:.4f}")
    
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions, target_names=['NEGATIVE', 'POSITIVE'])
        
        return avg_loss, accuracy, report
    
    def predict(self, text):
        self.model.eval()
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        
        return "POSITIVE" if prediction == 1 else "NEGATIVE"

if __name__ == "__main__":
    # Création de l'analyseur
    print("Chargement du modèle CamemBERT...")
    analyzer = BERTSentimentAnalyzer()
    
    # Entraînement du modèle
    print("\nDébut de l'entraînement...")
    analyzer.train()
    
    # Exemples de prédictions
    test_phrases = [
        "Depuis ce matin votre application ne marche pas, je n'arrive pas à déverrouiller ma voiture.",
        "J'ai adoré la prestation, tout s'est parfaitement déroulé !",
        "Le service est lent et inefficace.",
        "Je recommande vivement ce produit, il est génial !",
        "La qualité est moyenne, pas terrible pour le prix",
        "Très bon rapport qualité-prix, je suis satisfait de mon achat"
    ]
    
    print("\n" + "="*50)
    print("Exemples de prédictions:")
    print("="*50)
    
    for phrase in test_phrases:
        sentiment = analyzer.predict(phrase)
        print(f"\nTexte: {phrase}")
        print(f"Sentiment: {sentiment}")
        print("-" * 50)
