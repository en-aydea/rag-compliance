# src/main.py
import asyncio
import logging
from sqlalchemy.orm import sessionmaker
from src.models import SessionLocal, CallInput, CallComplianceAnalysis
from src.compliance_chain import run_compliance_analysis # Ana RAG akışımız

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Her döngüde kaç çağrı işlenecek
BATCH_SIZE = 5 

async def process_batch(db_session, call_batch):
    """
    Bir grup çağrıyı asenkron olarak işler ve sonuçları veritabanına yazar.
    """
    
    tasks = []
    for call in call_batch:
        # Her çağrı için run_compliance_analysis fonksiyonunu bir görev (task) olarak ekle
        tasks.append(run_compliance_analysis(call.transcript))

    log.info(f"{len(call_batch)} adet çağrı için analiz görevleri başlatılıyor...")
    
    # Tüm analiz görevlerini paralel olarak çalıştır
    # results_list, her bir çağrı için bir 'analiz sonucu listesi' dönecek
    # Yani -> [ [segment_result_1, segment_result_2], [segment_result_1], ... ]
    try:
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        log.error(f"Asyncio.gather hatası (kritik): {e}")
        return

    log.info("Tüm analiz görevleri tamamlandı. Sonuçlar veritabanına yazılıyor...")
    
    # Sonuçları işle ve veritabanına yaz
    for i, result_or_exception in enumerate(results_list):
        call = call_batch[i]
        
        if isinstance(result_or_exception, Exception):
            # Eğer analiz sırasında bir hata oluştuysa (örn: LLM hatası)
            log.error(f"Çağrı ID {call.call_id} işlenirken hata oluştu: {result_or_exception}")
            call.status = "failed"
        
        elif not result_or_exception:
            # Analiz başarılı oldu ama hiç segment bulunamadı
            log.warning(f"Çağrı ID {call.call_id} için analiz edilecek segment bulunamadı.")
            call.status = "processed_no_segment"
            
        else:
            # Analiz başarılı ve segmentler bulundu
            try:
                # result_or_exception = [segment_1_dict, segment_2_dict, ...]
                for segment_data in result_or_exception:
                    new_analysis_output = CallComplianceAnalysis(
                        input_call_id=call.id,
                        **segment_data # Sözlükteki tüm verileri modele eşle
                    )
                    db_session.add(new_analysis_output)
                
                call.status = "processed"
                log.info(f"Çağrı ID {call.call_id} için {len(result_or_exception)} segment DB'ye eklendi.")
                
            except Exception as e:
                log.error(f"Çağrı ID {call.call_id} sonuçları DB'ye yazılırken hata: {e}")
                call.status = "failed_writing_db"

    try:
        db_session.commit()
    except Exception as e:
        log.error(f"Batch commit sırasında DB hatası: {e}")
        db_session.rollback()

def run_pipeline():
    """Ana BDDK Uyumluluk Pipeline'ı."""
    log.info("BDDK Uyumluluk Analiz Pipeline'ı Başlatılıyor...")
    
    db_session = SessionLocal()
    try:
        while True:
            log.info(f"'pending' statüsündeki {BATCH_SIZE} adet çağrı aranıyor...")
            
            call_batch = db_session.query(CallInput).filter(
                CallInput.status == "pending"
            ).limit(BATCH_SIZE).all()

            if not call_batch:
                log.info("İşlenecek yeni çağrı bulunamadı. Pipeline tamamlandı.")
                break
            
            log.info(f"{len(call_batch)} adet çağrı bulundu ve işleme alınıyor.")
            # Asenkron batch işleme fonksiyonunu çalıştır
            asyncio.run(process_batch(db_session, call_batch))

    except Exception as e:
        log.error(f"Pipeline'da kritik hata: {e}")
        db_session.rollback()
    finally:
        db_session.close()
        log.info("Veritabanı bağlantısı kapatıldı.")

if __name__ == "__main__":
    run_pipeline()