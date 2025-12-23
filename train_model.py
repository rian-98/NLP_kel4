import pandas as pd
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# 1. Inisialisasi Sastrawi (Syarat 5c)
print("Sedang memproses teks...")
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess_simple(text):
    return stemmer.stem(str(text).lower())

# 2. Load Data
try:
    df = pd.read_csv("dataset_emosi.csv")
    if df.empty:
        print("Error: Dataset kamu kosong!")
        exit()
except Exception as e:
    print(f"Error membaca file: {e}")
    exit()

# 3. Label Encoding (Syarat Output 6b)
le = LabelEncoder()
y = le.fit_transform(df['emosi'])

# 4. Feature Extraction: TF-IDF (Syarat 5d)
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['teks'].apply(preprocess_simple))

# 5. Model Training: SVM (Model 1)
model_svm = SVC(probability=True)
model_svm.fit(X, y)

# 6. PENYIMPANAN MODEL (Output Wajib 6b)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
    
with open('model_tfidf.pkl', 'wb') as f:
    pickle.dump((tfidf, model_svm), f)

print("✅ Berhasil! File 'model_tfidf.pkl' dan 'label_encoder.pkl' sudah muncul di folder.")