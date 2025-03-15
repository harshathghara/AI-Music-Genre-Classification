import numpy as np
import pandas as pd
import joblib

# Define CSV path
csv_path = "data/features_30_sec.csv"

# Load the trained model, label encoder, and scaler
model = joblib.load("models/genre_classifier.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
scaler = joblib.load("models/scaler.pkl")  # Load saved scaler

# Load dataset
df = pd.read_csv(csv_path)

# Drop the 'filename' column if it exists
if 'filename' in df.columns:
    df = df.drop(columns=['filename'])

# Extract feature names (excluding genre)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]  # True genre labels

# Ask user for a song index
index = int(input(f"Enter a song index (0-{len(df)-1}): "))

# Validate index
if index < 0 or index >= len(df):
    print("Invalid index. Please enter a valid number.")
else:
    # Extract features of the selected song
    song_features = np.array(X.iloc[index]).reshape(1, -1)

    # Apply the same feature scaling as in training
    song_features_scaled = scaler.transform(song_features)

    # Predict the genre
    predicted_label = model.predict(song_features_scaled)[0]
    predicted_genre = label_encoder.inverse_transform([predicted_label])[0]

    # Display results
    print(f"\nPredicted Genre: {predicted_genre}")
