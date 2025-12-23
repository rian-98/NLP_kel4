import streamlit as st
import pandas as pd
import pickle
import random
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# --- CONFIG HALAMAN ---
st.set_page_config(page_title="Chatbot Konseling Siswa", page_icon="🎓")

# Custom CSS Dark Mode
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: white; }
    .stChatInput textarea { background-color: #262730 !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD RESOURCES (DATA & MODEL) ---
@st.cache_resource
def load_all():
    df = pd.read_csv("dataset_emosi.csv")
    with open('model_tfidf.pkl', 'rb') as f:
        tfidf, model = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    stemmer = StemmerFactory().create_stemmer()
    return df, tfidf, model, le, stemmer

df, tfidf, model, le, stemmer = load_all()

# --- UI HEADER ---
st.title("🎓 Chatbot Konseling Siswa")
st.caption("Aplikasi NLP Online - SVM & TF-IDF with Similarity Threshold")
st.write("---")

# --- SESSION STATE CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Halo! Aku asisten konselingmu. Ada yang ingin kamu ceritakan hari ini?"}]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- PROSES INPUT USER ---
if prompt := st.chat_input("Ketik pesanmu di sini..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # A. Preprocessing
    clean_text = stemmer.stem(prompt.lower())
    vec = tfidf.transform([clean_text])
    
    # B. Prediksi Emosi
    pred_idx = model.predict(vec)
    emosi_label = le.inverse_transform(pred_idx)[0]
    
    # C. Retrieval dengan Threshold (Agar Konsisten)
    all_vecs = tfidf.transform(df['teks'].astype(str))
    similarities = cosine_similarity(vec, all_vecs).flatten()
    max_sim = similarities.max()

    # Logika Jawaban: Hanya menjawab jika kemiripan > 0.2
    if max_sim < 0.2:
        jawaban_final = "Maaf, aku tidak yakin memahami perasaanmu. Bisa ceritakan lebih detail?"
    else:
        # Ambil 3 jawaban terbaik untuk variasi
        top_k = similarities.argsort()[-3:]
        chosen_idx = random.choice(top_k)
        jawaban_final = df.iloc[chosen_idx]['jawaban']

    full_response = f"{jawaban_final} \n\n (Emosi: **{emosi_label}**)"
    
    with st.chat_message("assistant"):
        st.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})