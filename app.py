import streamlit as st
import pandas as pd
import pickle
import random
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# --- 1. KONFIGURASI HALAMAN & TEMA HITAM ---
st.set_page_config(
    page_title="Chatbot Konseling Siswa", 
    page_icon="🎓", 
    layout="centered"
)

# Style Custom Dark Mode (Syarat Output 6d)
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    .stChatInput textarea {
        background-color: #262730 !important;
        color: white !important;
    }
    h1, h2, h3, p, span {
        color: #FFFFFF !important;
    }
    hr {
        border-color: #31333F;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD RESOURCE (DATA & MODEL) ---
# Menggunakan cache agar aplikasi cepat saat dibuka
@st.cache_resource
def load_all():
    # Load Dataset
    df = pd.read_csv("dataset_emosi.csv")
    
    # Load Model & Vectorizer (Syarat 6b)
    with open('model_tfidf.pkl', 'rb') as f:
        tfidf, model = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
        
    # Inisialisasi Sastrawi (Syarat 5c)
    stemmer = StemmerFactory().create_stemmer()
    
    return df, tfidf, model, le, stemmer

# Memanggil resource
try:
    df, tfidf, model, le, stemmer = load_all()
except Exception as e:
    st.error(f"Gagal memuat file model/dataset. Pastikan file sudah di-upload ke GitHub. Error: {e}")
    st.stop()

# --- 3. UI HEADER ---
st.title("🎓 Chatbot Konseling Siswa")
st.caption("Aplikasi NLP Online - Klasifikasi Emosi & Retrieval Response")
st.write("---")

# --- 4. RIWAYAT CHAT (Continue Chat) ---
# Syarat 6d: Demo Deployment yang interaktif
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo! Aku asisten konselingmu. Ceritakan masalahmu tentang sekolah, pacar, atau biaya pendidikan."}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. LOGIKA PREDIKSI & VARIATION ---
if prompt := st.chat_input("Ketik pesanmu di sini..."):
    # Simpan & Tampilkan pesan user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # A. Preprocessing (Syarat 5c)
    clean_text = stemmer.stem(prompt.lower())
    
    # B. Feature Extraction (TF-IDF)
    vec = tfidf.transform([clean_text])
    
    # C. Prediksi Emosi (SVM)
    pred_idx = model.predict(vec)
    emosi_label = le.inverse_transform(pred_idx)[0]
    
    # D. Retrieval Response dengan Variasi (Top-3 Similarity)
    # Mencari kemiripan input dengan seluruh baris di dataset
    all_vecs = tfidf.transform(df['teks'].astype(str))
    similarities = cosine_similarity(vec, all_vecs).flatten()
    
    # Mengambil 3 jawaban terbaik agar hasil bervariasi
    top_k_indices = similarities.argsort()[-3:] 
    chosen_index = random.choice(top_k_indices)
    
    jawaban_final = df.iloc[chosen_index]['jawaban']

    # Gabungkan hasil
    full_response = f"{jawaban_final} \n\n (Terdeteksi emosi: **{emosi_label}**)"

    # E. Tampilkan & Simpan Respon Bot
    with st.chat_message("assistant"):
        st.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})