import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# --- 1. CONFIG & CSS ---
st.set_page_config(page_title="Chatbot Konseling", page_icon="🎓")
st.markdown("<style>.stApp {background-color: #0E1117; color: white;}</style>", unsafe_allow_html=True)

# --- 2. LOAD DATA & MODEL LANGSUNG (Tanpa API Eksternal) ---
@st.cache_resource
def load_resources():
    df = pd.read_csv("dataset_emosi.csv")
    with open('model_tfidf.pkl', 'rb') as f:
        tfidf, model = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    stemmer = StemmerFactory().create_stemmer()
    return df, tfidf, model, le, stemmer

df, tfidf, model, le, stemmer = load_resources()

# --- 3. UI & CHAT HISTORY ---
st.title("🎓 Chatbot Konseling Siswa (Online)")
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Halo! Aku sudah online. Ada yang ingin kamu ceritakan?"}]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- 4. LOGIKA PREDIKSI ---
if prompt := st.chat_input("Ketik di sini..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Preprocessing & Predict
    clean_input = stemmer.stem(prompt.lower())
    vec = tfidf.transform([clean_input])
    
    # Prediksi Emosi
    res_idx = model.predict(vec)
    emosi = le.inverse_transform(res_idx)[0]
    
    # Cari Jawaban (Similarity)
    all_vecs = tfidf.transform(df['teks'].astype(str))
    sim = cosine_similarity(vec, all_vecs)
    jawaban = df.iloc[sim.argmax()]['jawaban']

    full_res = f"{jawaban} \n\n (Emosi: **{emosi}**)"
    
    with st.chat_message("assistant"):
        st.markdown(full_res)
    st.session_state.messages.append({"role": "assistant", "content": full_res})