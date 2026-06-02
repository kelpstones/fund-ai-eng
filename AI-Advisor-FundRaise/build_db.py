import os
import json
import chromadb
from google import genai
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY tidak ditemukan di .env!")
    exit()

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

print("Membaca data knowledge_base.json...")
try:
    with open('knowledge_base.json', 'r', encoding='utf-8') as file:
        kb_data = json.load(file)
except FileNotFoundError:
    print("ERROR: File knowledge_base.json tidak ditemukan!")
    exit()

print("Menghubungkan ke ChromaDB...")
chroma_client = chromadb.PersistentClient(path="./fundraise_vectordb")
collection = chroma_client.get_or_create_collection(name="kb_fundraise")

print("Mulai proses embedding...")
for i, item in enumerate(kb_data):
    doc_id = f"doc_{i}"
    teks = item["teks"]
    kategori = item["kategori"]
    
    # Cek agar tidak duplikat
    existing = collection.get(ids=[doc_id])
    if not existing['ids']:
        print(f"Embedding dokumen {i+1}/{len(kb_data)}...")
        res = gemini_client.models.embed_content(model="gemini-embedding-2", contents=teks)
        collection.add(
            embeddings=[res.embeddings[0].values],
            documents=[teks],
            metadatas=[{"kategori": kategori}],
            ids=[doc_id]
        )
    else:
        print(f"Dokumen {i+1} sudah ada, di-skip.")

print("\nSUKSES! Database vektor selesai dibangun.")