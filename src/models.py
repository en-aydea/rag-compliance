# src/models.py
import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
from pydantic import BaseModel, Field
from typing import List, Optional

# =================================================================
# VERİTABANI (SQLALCHEMY) AYARLARI
# =================================================================

# Analiz edilecek çağrıların ve sonuçların tutulacağı lokal veritabanı
DATABASE_URL = "sqlite:///./bank_compliance.db"

Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class CallInput(Base):
    """Gelen çağrı transkriptlerinin tutulduğu kaynak tablo."""
    __tablename__ = "calls_input"
    id = Column(Integer, primary_key=True, index=True)
    call_id = Column(String, unique=True, index=True) # Çağrıya ait benzersiz ID
    transcript = Column(Text, nullable=False)
    status = Column(String, default="pending") # (pending, processed, failed)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class CallComplianceAnalysis(Base):
    """LLM tarafından yapılan BDDK uyumluluk analizinin sonuçları."""
    __tablename__ = "compliance_analysis_output"
    id = Column(Integer, primary_key=True, index=True)
    
    # Hangi çağrıya ait olduğu
    input_call_id = Column(Integer, ForeignKey("calls_input.id"), index=True)
    
    # Transkriptin hangi segmentine ait olduğu (örn: 1., 2. Soru-Cevap bloğu)
    segment_index = Column(Integer, nullable=False)
    
    # LLM 1 (Segmentasyon) Çıktıları
    customer_query = Column(Text, nullable=False)
    agent_response = Column(Text, nullable=False)
    
    # RAG Sonucu
    rag_context = Column(Text, nullable=True) # RAG'dan gelen en alakalı mevzuat metni
    
    # LLM 2 (Analiz) Çıktıları
    violation_detected = Column(Boolean, nullable=True) # İhlal var mı?
    omission_detected = Column(Boolean, nullable=True)  # Eksik bilgi var mı?
    analysis = Column(Text, nullable=True)              # Denetçi analizi
    suggestion = Column(Text, nullable=True)            # Temsilci için öneri
    
    processed_at = Column(DateTime(timezone=True), server_default=func.now())

def create_db_and_tables():
    """Veritabanı ve tabloları oluşturur."""
    Base.metadata.create_all(bind=engine)

# =================================================================
# LLM ÇIKTI (PYDANTIC) MODELLERİ
# =================================================================

class Segment(BaseModel):
    """
    LLM 1 (Segmentasyon) Çıktı Modeli.
    Transkriptteki bir 'Müşteri Sorusu' ve ona verilen 'Temsilci Cevabı' bloğu.
    """
    customer_query: str = Field(description="Müşterinin sorduğu spesifik soru veya dile getirdiği talep.")
    agent_response: str = Field(description="Temsilcinin, müşterinin o sorusuna/talebine verdiği spesifik cevap.")

class TranscriptSegments(BaseModel):
    """Transkriptteki tüm Soru-Cevap segmentlerinin listesi."""
    segments: List[Segment]

class AnalysisResult(BaseModel):
    """
    LLM 2 (Analiz) Çıktı Modeli.
    Bir segmentin BDDK mevzuatına göre analiz sonucu.
    """
    violation_detected: bool = Field(description="Temsilcinin cevabı, sağlanan mevzuata göre net bir İHLAL içeriyor mu? (Evet=true, Hayır=false)")
    omission_detected: bool = Field(description="Temsilci, müşterinin sorusuyla ilgili mevzuatın gerektirdiği kritik bir bilgiyi (örn: faiz tavanı, vade sınırı) müşteriye iletmeyi UNUTTU MU veya EKSİK BİLGİ verdi mi? (Evet=true, Hayır=false)")
    analysis: str = Field(description="Bulgularını (ihlal veya eksiklik varsa) açıklayan 1-2 cümlelik kısa denetçi raporu. Bulg yoksa 'Mevzuata uygundur' yaz.")
    suggestion: Optional[str] = Field(default=None, description="Bir ihlal veya eksiklik varsa, temsilcinin bu durumda tam olarak ne demesi gerektiğini belirten 'doğru' cevap önerisi.")