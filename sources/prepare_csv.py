import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Définir les chemins vers les répertoires audio et texte
audio_dir = 'data/audio'
text_dir = 'data/rapports'

# Créer une liste pour stocker les données
data = []

# Boucler sur chaque fichier dans le répertoire audio
for audio_file in sorted(os.listdir(audio_dir)):
    if audio_file.endswith('.wav'):  # Supposons que les fichiers audio sont au format .wav
        # Construire le chemin du fichier texte correspondant
        text_file = os.path.splitext(audio_file)[0] + '.txt'
        text_file_path = os.path.join(text_dir, text_file)
        
        # Lire la transcription si le fichier texte existe
        if os.path.exists(text_file_path):
            with open(text_file_path, 'r', encoding='utf-8') as f:
                transcription = f.read().strip()
        
            # Ajouter les informations à la liste des données
            data.append({
                'audio': os.path.join(audio_dir, audio_file),
                'sentence': transcription
            })

# Convertir la liste en DataFrame
df = pd.DataFrame(data)

# Diviser les données en 80% d'entraînement et 20% de test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Sauvegarder les DataFrames dans des fichiers CSV séparés pour l'entraînement et le test
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

print("Les fichiers CSV pour l'entraînement et le test ont été créés avec succès.")
