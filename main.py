import warnings
warnings.simplefilter('ignore')

import os
import pandas as pd
import unidecode
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Téléchargement des ressources NLTK nécessaires
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('french') + list(string.punctuation))
    
    def preprocess_text(self, text):
        # Convertir en minuscules
        text = text.lower()
        # Supprimer la ponctuation
        text = ''.join([char for char in text if char not in string.punctuation])
        # Tokenization
        tokens = nltk.word_tokenize(text, language='french')
        # Supprimer les stopwords et lemmatiser
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)

class SentimentAnalyzer:
    def __init__(self):
        # Initialisation
        self.preprocessor = TextPreprocessor()
        dataset_path = os.path.join('model', 'dataset.txt')
        separator = "   "  # 3 espaces comme séparateur
        
        # Chargement des données
        self.dataset = pd.read_csv(dataset_path, names=['sentence', 'label'], sep=separator)
        
        # Prétraitement des textes
        self.dataset['processed_text'] = self.dataset['sentence'].apply(
            lambda x: self.preprocessor.preprocess_text(str(x)))
        
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = None
        self.score = None
        self.train()

    def train(self):
        # Séparation des données
        X = self.vectorizer.fit_transform(self.dataset['processed_text'])
        y = self.dataset['label'].values

        # Division des données (80% entraînement, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        # Initialisation du modèle avec des paramètres optimisés
        self.model = XGBClassifier(
            max_depth=4,
            n_estimators=100,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        # Validation croisée
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        print(f"Précision moyenne en validation croisée: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
        
        # Entraînement final
        self.model.fit(X_train, y_train)
        
        # Évaluation sur l'ensemble de test
        y_pred = self.model.predict(X_test)
        self.score = accuracy_score(y_test, y_pred)
        print(f"\nPrécision sur l'ensemble de test: {self.score*100:.2f}%")
        print("\nRapport de classification:")
        print(classification_report(y_test, y_pred, target_names=['NEGATIVE', 'POSITIVE']))

    def predict(self, text):
        """
        Prédit le sentiment d'un texte donné
        Retourne 'POSITIVE' ou 'NEGATIVE'
        """
        # Prétraitement du texte
        processed_text = self.preprocessor.preprocess_text(text)
        # Vectorisation
        text_vectorized = self.vectorizer.transform([processed_text])
        # Prédiction
        prediction = self.model.predict(text_vectorized)
        
        # Conversion du résultat en texte lisible
        return "POSITIVE" if prediction[0] == 1 else "NEGATIVE"

if __name__ == "__main__":
    # Création de l'analyseur
    print("Entraînement du modèle en cours...\n")
    analyzer = SentimentAnalyzer()
    
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
