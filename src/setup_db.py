# src/setup_db.py
import pandas as pd
import logging
from src.models import engine, create_db_and_tables, CallInput, SessionLocal

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Ayarlar (Kullanıcıdan alınan bilgilere göre - DÜZELTİLDİ)
XLSX_PATH = "data/new_calls.xlsx" # Düzeltme: .xlsx yolu
TRANSCRIPT_COLUMN_NAME = "Transkript" 
CALL_ID_COLUMN_NAME = "Çağrı ID" 

def load_xlsx_to_db():
    log.info("Veritabanı ve tablolar oluşturuluyor...")
    # models.py'dan fonksiyonu çağır
    create_db_and_tables()
    
    log.info(f"'{XLSX_PATH}' dosyasından veriler okunuyor...")
    try:
        # --- DÜZELTME BURADA ---
        # pd.read_csv yerine pd.read_excel kullanıyoruz
        df = pd.read_excel(XLSX_PATH)
    except FileNotFoundError:
        log.error(f"HATA: '{XLSX_PATH}' dosyası bulunamadı.")
        return
    except Exception as e:
        log.error(f"XLSX okuma hatası: {e}")
        log.error("İpucu: 'pip install openpyxl' komutunu çalıştırdınız mı?")
        return

    # Veritabanı oturumu başlatma
    db_session = SessionLocal()

    log.info("Transkriptler 'calls_input' tablosuna yükleniyor...")
    count = 0
    try:
        for index, row in df.iterrows():
            call_id = str(row.get(CALL_ID_COLUMN_NAME, f"call_{index}"))
            transcript = row.get(TRANSCRIPT_COLUMN_NAME)

            if not transcript or pd.isna(transcript):
                log.warning(f"Satır {index} (ID: {call_id}) atlanıyor: Transkript boş.")
                continue

            # Bu call_id daha önce eklendi mi diye kontrol et
            exists = db_session.query(CallInput).filter_by(call_id=call_id).first()
            if not exists:
                new_call = CallInput(
                    call_id=call_id,
                    transcript=str(transcript),
                    status="pending" # Başlangıç durumu
                )
                db_session.add(new_call)
                count += 1
            
        db_session.commit()
        log.info(f"Başarıyla {count} adet yeni çağrı transkripti veritabanına eklendi.")
    except Exception as e:
        db_session.rollback()
        log.error(f"Veritabanına yazma hatası: {e}")
    finally:
        db_session.close()

if __name__ == "__main__":
    load_xlsx_to_db()