import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_preprocess_data(csv_path):
    # Load dataset
    df = pd.read_csv(csv_path)
    
    # Drop the filename column (not useful for training)
    df.drop(columns=["filename"], inplace=True)
    
    # Encode the genre labels into numerical values
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["label"])
    
    # Separate features and target labels
    X = df.drop(columns=["label"])  # Features
    y = df["label"]                 # Target labels
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into training and test sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, label_encoder

if __name__ == "__main__":
    csv_path = "data/features_30_sec.csv"  # Update path if needed
    X_train, X_test, y_train, y_test, label_encoder = load_preprocess_data(csv_path)
    print("Dataset loaded successfully! Shapes:")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
