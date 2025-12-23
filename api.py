from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

app = FastAPI()
stemmer = StemmerFactory().create_stemmer()

# Load data dan model yang sudah di-train
df = pd.read_csv("dataset_emosi.csv")
with open('model_tfidf.pkl', 'rb') as f:
    tfidf, model = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

class InputData(BaseModel):
    teks: str

@app.post("/predict")
def predict(data: InputData):
    # 1. Preprocessing input
    clean_input = stemmer.stem(data.teks.lower())
    input_vec = tfidf.transform([clean_input])
    
    # 2. Prediksi Emosi menggunakan Model
    res_idx = model.predict(input_vec)
    emosi_label = le.inverse_transform(res_idx)[0]
    
    # 3. Cari JAWABAN di dataset yang paling mirip (Similarity Search)
    # Transform seluruh kolom teks di dataset menjadi vektor TF-IDF
    all_text_vecs = tfidf.transform(df['teks'].astype(str))
    
    # Hitung kemiripan antara input user dengan semua baris di dataset
    similarities = cosine_similarity(input_vec, all_text_vecs)
    index_terdekat = similarities.argmax()
    
    # Ambil jawaban asli dari kolom 'jawaban' di dataset
    jawaban_final = df.iloc[index_terdekat]['jawaban']
    
    return {
        "emosi": emosi_label,
        "jawaban": jawaban_final
    }