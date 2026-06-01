import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types
from prometheus_fastapi_instrumentator import Instrumentator
import chromadb

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI(title="FundRaise Chatbot API - PRODUCTION")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
Instrumentator().instrument(app).expose(app)

# Koneksi ke ChromaDB
try:
    chroma_client = chromadb.PersistentClient(path="./fundraise_vectordb")
    collection = chroma_client.get_collection(name="kb_fundraise")
except Exception as e:
    print(f"WARNING: Gagal memuat ChromaDB: {e}")
    collection = None

# Konfigurasi Gemini API & Dual-Persona
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    
    bot_persona = (
        "Anda adalah AI Multi-Fungsi resmi untuk platform FundRaise. "
        "Platform ini dibangun oleh tim capstone CC26-PSU027 (Danendra, Azra, Aldi, Yazid, Andika, Adam).\n\n"
        "Anda memiliki DUA peran utama:\n"
        "PERAN 1: CUSTOMER SERVICE FUNDRAISE\n"
        "Jika pengguna bertanya tentang platform FundRaise, metrik platform, aturan, atau cara menggunakan fitur, Anda WAJIB menjawab HANYA berdasarkan 'Konteks Data Internal'. DILARANG mengarang fitur.\n\n"
        "PERAN 2: BUSINESS & INVESTMENT ADVISOR UMUM\n"
        "Jika pengguna meminta nasihat bisnis/operasional, atau strategi investasi dan analisis risiko, abaikan Konteks Internal dan gunakan wawasan ahli Anda.\n\n"
        "GAYA BAHASA & FORMAT JAWABAN:\n"
        "1. FORMAT FLEKSIBEL: Gunakan list (bernomor/bullet) untuk panduan langkah, tips, atau metrik. Gunakan paragraf biasa untuk obrolan santai.\n"
        "2. Bersikaplah profesional, ramah, dan RINGKAS.\n\n"
        "ATURAN FORMAT SAPAAN AWAL:\n"
        "HANYA JIKA pengguna memberikan sapaan awal saja (seperti 'Halo') ATAU murni menanyakan fungsi Anda tanpa pertanyaan lain, Anda WAJIB merespon dengan teks ini:\n\n"
        "\"Halo! Saya adalah AI Multi-Fungsi resmi dari platform FundRaise, yang dikembangkan oleh tim CC26-PSU027.\n\n"
        "Saya siap membantu Anda dalam dua hal utama:\n"
        "1. **Customer Service FundRaise**: Memberikan informasi mengenai fitur platform, panduan penggunaan, aturan, dan metrik yang ada di FundRaise.\n"
        "2. **Business & Investment Advisor**: Memberikan nasihat bisnis, strategi pemasaran, atau tips operasional bagi Pemilik UMKM, serta memberikan analisis risiko dan strategi investasi bagi Investor.\n\n"
        "Ada yang bisa saya bantu hari ini? Apakah Anda ingin bertanya mengenai cara kerja platform kami atau membutuhkan saran terkait bisnis dan investasi Anda?\"\n\n"
        "PENTING: JIKA pengguna langsung menanyakan hal spesifik (seperti tips bisnis, metrik investasi, atau cara kerja platform), JANGAN tampilkan teks sapaan di atas. LANGSUNG jawab pertanyaannya saja tanpa basa-basi.\n\n"
        "ATURAN RAHASIA KETAT:\n"
        "DILARANG memberikan rekomendasi nama UMKM spesifik. HANYA sebutkan aturan penolakan ini dan arahkan ke fitur 'Rekomendasi Pintar' JIKA user secara eksplisit meminta rekomendasi nama UMKM."
    )
else:
    gemini_client = None
    print("WARNING: GEMINI_API_KEY belum di-set di .env!")

# Fungsi Pencarian Konteks RAG dengan Distance Filter
def retrieve_relevant_info(query: str, top_k: int = 2) -> str:
    if collection is None:
        return "Tidak ada panduan internal yang cocok. (Database Vector tidak aktif)"
    
    try:
        query_response = gemini_client.models.embed_content(model="gemini-embedding-2", contents=query)
        query_vec = query_response.embeddings[0].values
        
        results = collection.query(
            query_embeddings=[query_vec], 
            n_results=top_k,
            include=['documents', 'distances']
        )
        
        if results['documents'] and len(results['documents'][0]) > 0:
            if results['distances'][0][0] > 0.80:
                return "Tidak ada panduan internal yang cocok."
            return "\n".join(results['documents'][0])
            
    except Exception as e:
        print(f"WARNING: Gagal query ke VectorDB: {e}")
        
    return "Tidak ada panduan internal yang cocok."

# Skema Input cuma butuh Chat
class ChatRequest(BaseModel):
    message: str

# Endpoint AI Chatbot
@app.post("/chat")
def chat_with_advisor(request: ChatRequest):
    if gemini_client is None:
        raise HTTPException(
            status_code=500, 
            detail="Gemini API belum dikonfigurasi. Cek GEMINI_API_KEY di .env"
        )

    try:
        contekan_data = retrieve_relevant_info(request.message)
        
        rag_prompt = (
            f"Pertanyaan User: {request.message}\n\n"
            f"=== KONTEKS DATA INTERNAL FUNDRAISE ===\n"
            f"{contekan_data}\n"
            f"=======================================\n\n"
            f"INSTRUKSI: Tentukan apakah user bertanya soal teknis/aturan FundRaise (Peran 1), ATAU meminta nasihat bisnis umum (Peran 2). Jawablah sesuai porsi peran Anda."
        )

        chat_response = gemini_client.models.generate_content(
            model="gemini-3-flash-preview", 
            contents=[rag_prompt],
            config=types.GenerateContentConfig(
                system_instruction=bot_persona,
                temperature=0.3
            )
        )
        
        return {
            "status": "success",
            "reply": chat_response.text
        }
    
    except Exception as e:
        error_msg = str(e)
        print(f"Detail error: {error_msg}")
        
        if "429" in error_msg or "Quota exceeded" in error_msg:
            raise HTTPException(
                status_code=429, 
                detail="Mohon maaf, kuota bertanya AI Advisor sedang habis. Silakan coba lagi nanti."
            )
            
        raise HTTPException(
            status_code=500, 
            detail=f"AI Advisor sedang gangguan: {error_msg}"
        )

# Endpoint Health Update
@app.get("/health")
def health_check():
    services_status = {
        "gemini_api": gemini_client is not None,
        "chromadb": collection is not None
    }
    
    if not all(services_status.values()):
        raise HTTPException(
            status_code=503, 
            detail={
                "status": "error",
                "message": "Chatbot API is running, but some essential services failed to load.",
                "services": services_status
            }
        )
    
    return {
        "status": "success",
        "message": "FundRaise Chatbot API is healthy and ready to rock! 🚀",
        "services_loaded": True
    }