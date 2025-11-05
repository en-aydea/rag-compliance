# src/test_single_call.py
import asyncio
import logging
import json
import time

from src.models import SessionLocal, CallInput
# Ana RAG akışımızı (orkestratör) import ediyoruz
from src.compliance_chain import run_compliance_analysis

# --- AYAR ---
# Veritabanından (calls_input tablosu) test etmek istediğiniz çağrının 'id'sini girin.
# Bu 'Çağrı ID' (metin) değil, tablodaki 'id' (otomatik artan sayı) olmalıdır.
TEST_CALL_ID = 163
# -----------

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

async def run_single_test_async():
    """
    Belirlenen tek bir çağrı ID'si için tam "Çift Aşamalı RAG Uyumluluk Analizi" 
    akışını çalıştırır ve sonucu konsola basar.
    """
    log.info(f"Tekil Çağrı Testi Başlatılıyor (Çağrı ID: {TEST_CALL_ID})...")
    
    db_session = SessionLocal()
    
    try:
        # Adım 1: Test çağrısını veritabanından al
        test_call = db_session.query(CallInput).filter(CallInput.id == TEST_CALL_ID).first()
        if not test_call:
            log.error(f"HATA: ID'si {TEST_CALL_ID} olan çağrı veritabanında bulunamadı.")
            log.error("Lütfen önce 'python src/setup_db.py' komutunu çalıştırdığınızdan emin olun.")
            return
            
        log.info(f"Test transkripti (Call ID: {test_call.call_id}) başarıyla alındı.")
        full_transcript = test_call.transcript
        
    except Exception as e:
        log.error(f"Veritabanından çağrı okunurken hata: {e}")
        return
    finally:
        db_session.close()

    log.info("Uyumluluk Analiz Zinciri ('run_compliance_analysis') başlatılıyor...")
    start_time = time.time()

    try:
        # Adım 2: Ana RAG Akışını (Orkestratör) Çalıştır
        # Bu fonksiyon tüm adımları (Segmentasyon -> RAG -> Analiz) içerir.
        analysis_results = await run_compliance_analysis(full_transcript)
        
        end_time = time.time()
        log.info(f"Çağrı {end_time - start_time:.2f} saniyede başarıyla işlendi.")

        # Adım 3: Sonuçları JSON olarak formatla ve yazdır
        if not analysis_results:
            log.warning("Analiz tamamlandı ancak bu transkriptte BDDK ile ilgili bir segment bulunamadı.")
            return

        log.info(f"---------- {len(analysis_results)} ADET SEGMENT İÇİN ANALİZ SONUCU ----------")
        
        # Sonuçları (sözlük listesi) okunabilir JSON formatına çevir
        result_json = json.dumps(
            analysis_results, 
            indent=2, 
            ensure_ascii=False # Türkçe karakterleri koru
        )
        print(result_json)
        
        log.info("----------------------------------------------------------")

    except Exception as e:
        log.error(f"Tekil çağrı analizi sırasında bir hata oluştu: {e}")

if __name__ == "__main__":
    # Asenkron test fonksiyonunu çalıştırmak için
    asyncio.run(run_single_test_async())