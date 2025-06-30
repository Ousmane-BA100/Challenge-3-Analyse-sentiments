import os
from pymongo import MongoClient
from dotenv import load_dotenv

def import_data_to_mongodb():
    # Charger les variables d'environnement
    load_dotenv()
    
    # Connexion à MongoDB
    client = MongoClient(os.getenv('MONGO_URI'))
    db = client[os.getenv('MONGO_DB')]
    collection = db[os.getenv('MONGO_COLLECTION')]
    
    # Vider la collection si elle existe
    collection.drop()
    
    # Lire les données du fichier
    data = []
    with open('model/dataset.txt', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    text, label = line.rsplit('   ', 1)
                    data.append({
                        'text': text.strip(),
                        'label': int(label.strip())
                    })
                except ValueError:
                    print(f"Ligne ignorée (format incorrect): {line.strip()}")
    
    # Insérer les données dans MongoDB
    if data:
        result = collection.insert_many(data)
        print(f"{len(result.inserted_ids)} documents insérés avec succès dans MongoDB.")
    
    # Aperçu des données insérées
    print("\nAperçu des données dans la collection:")
    for doc in collection.find().limit(5):
        print(doc)

if __name__ == "__main__":
    import_data_to_mongodb()
