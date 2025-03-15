import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

# Load dataset
csv_path = "data/features_30_sec.csv"
df = pd.read_csv(csv_path)

# Extract features and labels
X = df.iloc[:, 1:-1].values  # Exclude filename and label
y = df.iloc[:, -1].values    # Genre labels

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape for CNN input (samples, features, 1)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define CNN model
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Conv1D(128, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
print("Training CNN model...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Model Accuracy: {test_acc:.4f}")

# Save model
model.save("models/genre_classifier_cnn.h5")
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("CNN model training completed and saved successfully!")
