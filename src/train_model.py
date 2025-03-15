import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from load_data import load_preprocess_data

# Load the dataset
csv_path = "data/features_30_sec.csv"
X_train, X_test, y_train, y_test, label_encoder = load_preprocess_data(csv_path)

# Apply feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize SVM model with better hyperparameters
svm_model = SVC(kernel="rbf", C=5, gamma="scale")

# Train the model
print("Training the SVM model...")
svm_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = svm_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save the trained model, label encoder, and scaler
joblib.dump(svm_model, "models/genre_classifier.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")
joblib.dump(scaler, "models/scaler.pkl")  # Save the scaler
