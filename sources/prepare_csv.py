import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Define the paths
audio_dir = 'data/audio'
text_dir = 'data/rapports'

# Create a list to hold the data
data = []

# Loop over each file in the audio directory
for audio_file in sorted(os.listdir(audio_dir)):
    if audio_file.endswith('.wav'):  # Assuming the audio files are in .wav format
        # Construct the corresponding text file path
        text_file = os.path.splitext(audio_file)[0] + '.txt'
        text_file_path = os.path.join(text_dir, text_file)
        
        # Read the transcription text if the file exists
        if os.path.exists(text_file_path):
            with open(text_file_path, 'r', encoding='utf-8') as f:
                transcription = f.read().strip()
        
            # Add the information to the data list
            data.append({
                'audio': os.path.join(audio_dir, audio_file),
                'sentence': transcription
            })

# Convert the list to a DataFrame
df = pd.DataFrame(data)

# Split the data into 80% train and 20% test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save the DataFrame to separate CSV files for train and test
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

print("Train and Test CSV files created successfully.")
