import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from sklearn.preprocessing import StandardScaler

# Load trained CNN model
model = load_model("models/genre_classifier_cnn.h5")

# Load label encoder
with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load dataset for feature scaling
csv_path = "data/features_30_sec.csv"
df = pd.read_csv(csv_path)

# Extract features (excluding filename and label)
X = df.iloc[:, 1:-1].values

# Standardize features
scaler = StandardScaler()
scaler.fit(X)  # Fit on the entire dataset

def predict_genre(feature_vector):
    """
    Predicts the genre based on the provided feature vector.
    
    Args:
    feature_vector (list or np.array): Extracted features of a song.

    Returns:
    str: Predicted genre
    """
    # Convert to NumPy array and reshape
    feature_vector = np.array(feature_vector).reshape(1, -1)

    # Standardize features
    feature_vector = scaler.transform(feature_vector)

    # Reshape for CNN input
    feature_vector = feature_vector.reshape(1, feature_vector.shape[1], 1)

    # Predict
    prediction = model.predict(feature_vector)
    predicted_label = np.argmax(prediction)

    return label_encoder.inverse_transform([predicted_label])[0]

if __name__ == "__main__":
    # Let the user select an index
    while True:
        try:
            index = int(input(f"Enter an index (0 to {len(df) - 1}): "))
            if 0 <= index < len(df):
                break
            else:
                print("Invalid index. Please enter a valid index.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Get features for the selected index
    selected_features = df.iloc[index, 1:-1].values  # Exclude filename & label
    predicted_genre = predict_genre(selected_features)
    
    print(f"Predicted Genre: {predicted_genre}")
