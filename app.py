import streamlit as st
import requests

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Chatbot Konseling Siswa", 
    page_icon="🎓", 
    layout="centered"
)

# --- 2. STYLE CUSTOM DARK MODE ---
# Syarat Output 6d: Demo Deployment
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

st.title("🎓 Chatbot Konseling Siswa")
st.caption("Respon chatbot ini diambil langsung dari dataset emosi.")
st.write("---")

# --- 3. INISIALISASI SESSION STATE (Agar chat berkelanjutan) ---
# Menyimpan riwayat agar percakapan tetap muncul di layar
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo! Aku asisten konselingmu. Ceritakan masalahmu, aku akan mencari jawaban terbaik dari dataku."}
    ]

# --- 4. MENAMPILKAN RIWAYAT CHAT ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. INPUT CHAT & LOGIKA PREDIKSI ---
if prompt := st.chat_input("Ketik pesanmu di sini..."):
    # Tampilkan pesan user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Memanggil Backend API (FastAPI)
    try:
        # Melakukan POST request ke API untuk mendapatkan emosi dan jawaban dataset
        response = requests.post(
            "http://127.0.0.1:8000/predict", 
            json={"teks": prompt},
            timeout=10
        )
        
        if response.status_code == 200:
            data_res = response.json()
            # Mengambil jawaban spesifik dari dataset (Retrieval-based)
            jawaban_dataset = data_res.get("jawaban")
            emosi = data_res.get("emosi")
            
            full_response = f"{jawaban_dataset} \n\n (Terdeteksi emosi: **{emosi}**)"
        else:
            full_response = "Maaf, sistem gagal mengambil jawaban dari dataset."

    except requests.exceptions.ConnectionError:
        full_response = "❌ **Koneksi Gagal**: Jalankan perintah `uvicorn api:app --reload` di terminal terlebih dahulu!"

    # Tampilkan respon bot
    with st.chat_message("assistant"):
        st.markdown(full_response)
    
    # Simpan ke riwayat agar chat terus berlanjut
    st.session_state.messages.append({"role": "assistant", "content": full_response})