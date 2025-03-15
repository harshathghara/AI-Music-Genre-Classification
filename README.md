# AI Music Genre Classification 🎵

# Overview

This project focuses on classifying music genres using machine learning models. Both Support Vector Machine (SVM) and Convolutional Neural Network (CNN) models have been implemented, achieving an accuracy of 74% each. The dataset used contains extracted numerical features from 30-second audio clips.

# 📂 Project Structure

AI_Music_Genre_Classification/
│── data/
│   ├── genres_original/      # Raw audio files (not included in repo)
│   ├── features_30_sec.csv   # Extracted features from audio
│── models/
│   ├── genre_classifier.pkl  # Trained SVM model
│   ├── genre_classifier_cnn.h5 # Trained CNN model
│   ├── label_encoder.pkl     # Label encoder for genre labels
│   ├── scaler.pkl            # Scaler for feature normalization
│── src/
│   ├── load_data.py          # Script for loading and preprocessing data
│   ├── train_model.py        # SVM model training script
│   ├── train_model_cnn.py    # CNN model training script
│── music_env/                # Virtual environment (not included in repo)
|── predict.py                # SVM model prediction script
├── predict_using_cnn.py      # CNN model prediction script
│── .gitignore                # Files to ignore in version control
│── requirements.txt          # Dependencies required to run the project
│── README.md                 # Project documentation

# 🔧 Setup Instructions

1️⃣ Clone the Repository

git clone https://github.com/yourusername/AI_Music_Genre_Classification.git
cd AI_Music_Genre_Classification

2️⃣ Create & Activate a Virtual Environment

python -m venv music_env
source music_env/bin/activate  # Mac/Linux
music_env\Scripts\activate     # Windows

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Train the Models

    To train the SVM model:
        python src/train_model.py   # svm model

    To train the CNN model:
        python src/train_model_cnn.py

5️⃣ Make Predictions

    Using SVM:
        python predict.py

    Using CNN:
        python predict_using_cnn.py

6️⃣ Run the Web Application

    python web_app.py

# 📊 Models Used

1️⃣ Support Vector Machine (SVM)

Uses extracted numerical features.

Provides a simple and effective classification approach.

Accuracy: 74%.

2️⃣ Convolutional Neural Network (CNN)

Learns deeper audio features for better classification.

Works well with spectrogram-based inputs.

Accuracy: 74%.

# 📄 Dataset Details

The dataset contains numerical feature representations of music clips from different genres:

Blues

Classical

Country

Disco

Hip-hop

Jazz

Metal

Pop

Reggae

Rock

# 🔥 Future Enhancements

Improve model accuracy through hyperparameter tuning.

Implement real-time audio processing for dynamic predictions.

Enhance the web interface for an interactive user experience.

# Contributors

Harsh Kumar (Lead Developer)


# License

This project is open-source and available under the MIT License.