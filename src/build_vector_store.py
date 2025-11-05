# src/build_vector_store.py
import os
import shutil
import logging
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    DOCUMENTS_PATH, 
    CHROMA_DB_PATH, 
    EMBEDDING_MODEL_NAME, 
    EMBEDDING_DEVICE
)

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def load_all_documents(directory_path: str):
    """
    Belirtilen klasördeki tüm PDF dosyalarını yükler ve birleştirir.
    """
    all_docs = []
    log.info(f"'{directory_path}' klasöründeki dokümanlar yükleniyor...")
    
    if not os.path.exists(directory_path):
        log.error(f"HATA: Doküman klasörü bulunamadı: {directory_path}")
        return []

    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            try:
                loader = PyMuPDFLoader(file_path)
                docs = loader.load()
                log.info(f" -> {filename} yüklendi ({len(docs)} sayfa).")
                all_docs.extend(docs)
            except Exception as e:
                log.warning(f"'{filename}' yüklenirken hata oluştu: {e}")
                
    return all_docs

def build_vector_store():
    """
    1. BDDK dokümanlarını yükler.
    2. Anlamsal olarak parçalara ayırır (chunking).
    3. Lokal (Hugging Face) model ile vektörleştirir.
    4. ChromaDB'ye kaydeder.
    """
    
    # 1. Dokümanları Yükle
    documents = load_all_documents(DOCUMENTS_PATH)
    if not documents:
        log.error("Hiç doküman yüklenemedi. İşlem durduruluyor.")
        return

    # 2. Parçalara Ayır (Chunking)
    # Hukuki metinler için paragrafları ve maddeleri korumak önemlidir.
    # Bu ayırıcı, önce çift satır boşluğuna (\n\n), sonra tek satır boşluğuna (\n)
    # göre bölmeyi dener. Bu, yönetmelik maddelerini bir arada tutmaya yardımcı olur.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Her parçanın maksimum boyutu (karakter)
        chunk_overlap=200,    # Parçalar arası bağlamı korumak için çakışma payı
        separators=["\n\n", "\n", " ", ""] # Bölme öncelik sırası
    )
    
    chunks = text_splitter.split_documents(documents)
    log.info(f"Toplam {len(documents)} sayfa, {len(chunks)} adet anlamsal parçaya (chunk) bölündü.")

    # 3. Lokal Embedding Modelini Hazırla
    log.info(f"Lokal embedding modeli '{EMBEDDING_MODEL_NAME}' yükleniyor...")
    log.info(f"Kullanılan cihaz: {EMBEDDING_DEVICE}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': EMBEDDING_DEVICE}
    )
    log.info("Embedding modeli başarıyla yüklendi.")

    # 4. Vektör Veritabanını Oluştur ve Kaydet
    # Önceki veritabanını (varsa) temizle
    if os.path.exists(CHROMA_DB_PATH):
        log.warning(f"Mevcut veritabanı '{CHROMA_DB_PATH}' siliniyor...")
        shutil.rmtree(CHROMA_DB_PATH)

    log.info("Vektör veritabanı oluşturuluyor ve dokümanlar vektörize ediliyor...")
    log.info("BU İŞLEM CPU ÜZERİNDE ZAMAN ALACAKTIR. LÜTFEN BEKLEYİN...")
    
    # Dokümanları vektörleştirir ve ChromaDB'ye kaydeder
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    
    log.info(f"Vektör veritabanı başarıyla '{CHROMA_DB_PATH}' adresine kaydedildi!")
    log.info(f"Toplam {vector_store._collection.count()} adet vektör eklendi.")

if __name__ == "__main__":
    build_vector_store()