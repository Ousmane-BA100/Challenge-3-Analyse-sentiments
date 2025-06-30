# Analyse de Sentiments en Français avec BERT et MongoDB

Ce projet implémente un modèle d'analyse de sentiments en français utilisant CamemBERT (version française de BERT) et stocke les données dans une base de données MongoDB conteneurisée avec Docker.

## Fonctionnalités

- Entraînement d'un modèle d'analyse de sentiments avec CamemBERT
- Stockage des données dans MongoDB
- API pour effectuer des prédictions
- Interface simple pour tester le modèle

## Prérequis

- Docker et Docker Compose
- Python 3.8+
- pip (gestionnaire de paquets Python)

## Installation

1. **Cloner le dépôt** :
   ```bash
   git clone https://github.com/Ousmane-BA100/Challenge-3-Analyse-sentiments.git
   cd Challenge-3-Analyse-sentiments
   ```

2. **Créer et activer un environnement virtuel (recommandé)** :
   ```bash
   # Sous Windows
   python -m venv .venv
   .venv\Scripts\activate
   
   # Sous macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

4. **Démarrer MongoDB avec Docker** :
   ```bash
   docker-compose up -d
   ```

5. **Importer les données dans MongoDB** :
   ```bash
   python mongo_importer.py
   ```

## Utilisation

### Entraîner le modèle
```bash
python bert_model.py
```

### Tester avec des exemples
Le script `bert_model.py` contient des exemples de prédiction qui s'exécutent après l'entraînement.

## Structure du projet

```
.
├── .env                    # Variables d'environnement
├── .gitignore              # Fichiers à ignorer par Git
├── README.md               # Ce fichier
├── bert_model.py           # Modèle BERT et logique d'entraînement
├── docker-compose.yml      # Configuration Docker pour MongoDB
├── mongo_importer.py       # Script d'importation des données
├── requirements.txt        # Dépendances Python
└── model/
    └── dataset.txt      # Données d'entraînement initiales
```

## Configuration

Les paramètres de configuration sont définis dans le fichier `.env` :

```env
# MongoDB Configuration
MONGO_URI=mongodb://root:example@localhost:27017/
MONGO_DB=sentiment_db
MONGO_COLLECTION=reviews
```

## Avertissements

- Les données d'entraînement fournies sont limitées. Pour de meilleures performances, envisagez d'utiliser un jeu de données plus important.
- L'entraînement sur CPU peut être lent. Pour de meilleures performances, utilisez un environnement avec GPU.

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.
