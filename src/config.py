# src/config.py
import os
from dotenv import load_dotenv

# .env dosyasındaki değişkenleri yükle
load_dotenv()

# =================================================================
# LLM AYARLARI (ANALİZ VE ÇIKARIM İÇİN)
# =================================================================
# Faz 2'deki analiz için OpenAI'nin güçlü modellerini kullanacağız.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY ortam değişkeni bulunamadı. .env dosyasını kontrol edin.")

LLM_MODEL = "gpt-4-turbo" # Veya "gpt-4o" - Uyumluluk analizi için güçlü bir model şart.

# =================================================================
# LOKAL EMBEDDING AYARLARI (RAG İÇİN)
# =================================================================
# CPU üzerinde çalışacak lokal modelimiz.
# https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DEVICE = "cpu" # 'cuda' (GPU) veya 'cpu'

# =================================================================
# DOSYA YOLLARI
# =================================================================
# BDDK dokümanlarının bulunduğu klasör
DOCUMENTS_PATH = "data/bddk_docs"

# Vektör veritabanının diske kaydedileceği yer
CHROMA_DB_PATH = "db/chroma_db"