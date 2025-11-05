# src/compliance_chain.py
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
from typing import List 

from src.config import (
    OPENAI_API_KEY, 
    LLM_MODEL, 
    CHROMA_DB_PATH, 
    EMBEDDING_MODEL_NAME, 
    EMBEDDING_DEVICE
)
# 'TranscriptSegments' ve 'AnalysisResult' modellerini models.py'dan alıyoruz
from src.models import TranscriptSegments, AnalysisResult 

log = logging.getLogger("compliance_chain")

# =================================================================
# 1. VEKTÖR VERİTABANI YÜKLEYİCİ
# =================================================================

def load_vector_store_retriever():
    """
    Diske kaydedilmiş ChromaDB'yi ve lokal embedding modelini yükler.
    Bir 'retriever' nesnesi döndürür.
    """
    log.info("Lokal embedding modeli yükleniyor...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': EMBEDDING_DEVICE}
    )
    
    log.info(f"ChromaDB '{CHROMA_DB_PATH}' adresinden yükleniyor...")
    vector_store = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings
    )
    
    # Retriever'ı k=3 olarak ayarlıyoruz.
    return vector_store.as_retriever(search_kwargs={"k": 3})

# =================================================================
# 2. ZİNCİR 1: TRANSKRİPT SEGMENTASYON ZİNCİRİ
# =================================================================

def create_segmentation_chain():
    """
    LLM Zincir 1: Ham transkripti alır, Soru-Cevap segmentlerine ayırır.
    """
    llm = ChatOpenAI(model=LLM_MODEL, openai_api_key=OPENAI_API_KEY, temperature=0)
    
    parser = PydanticOutputParser(pydantic_object=TranscriptSegments)
    
    prompt_template = """
    Senin görevin, bir banka çağrı merkezi transkriptini analiz etmektir.
    Transkripti, müşterinin bir soru sorduğu veya talepte bulunduğu ve temsilcinin 
    buna cevap verdiği mantıksal "Soru-Cevap" bloklarına ayırmalısın.
    
    Sadece BDDK veya bankacılık mevzuatıyla ilgili olabilecek (kredi, faiz, vade, 
    kart aidatı, borç yapılandırma vb.) Soru-Cevap segmentlerine odaklan.
    Kimlik doğrulama, 'nasılsınız', 'iyi günler' gibi alakasız diyalogları ATLA.

    Transkript:
    ---
    {transcript}
    ---
    
    Tespit ettiğin tüm ilgili segmentleri JSON formatında listele.
    {format_instructions}
    """
    
    prompt = ChatPromptTemplate.from_template(
        template=prompt_template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    return prompt | llm | parser

# =================================================================
# 3. YENİ ZİNCİR: SORGU ZENGİNLEŞTİRME (QUERY TRANSFORMATION)
# =================================================================

class SearchQuery(BaseModel):
    """BDDK Vektör Veritabanı için zenginleştirilmiş, resmi arama sorgusu."""
    search_query: str = Field(description="BDDK mevzuat veritabanında arama yapmak için optimize edilmiş, resmi ve anahtar kelime bakımından zengin sorgu.")

def create_query_transformation_chain():
    """
    LLM Zincir 1.5: Günlük konuşmayı alır, resmi bir RAG arama sorgusuna dönüştürür.
    """
    llm = ChatOpenAI(model=LLM_MODEL, openai_api_key=OPENAI_API_KEY, temperature=0)
    
    # Basit bir Pydantic parser yerine StrOutputParser da kullanabilirdik,
    # ancak Pydantic yapıya zorlayarak daha tutarlı sonuç alırız.
    parser = PydanticOutputParser(pydantic_object=SearchQuery)
    
    prompt_template = """
    Sen bir bankacılık uzmanısın. Görevin, bir çağrı merkezi diyaloğunu okuyup, 
    bu diyaloğun hangi BDDK mevzuatıyla ilgili olduğunu anlamak ve 
    vektör veritabanı araması için resmi bir arama sorgusu oluşturmaktır.

    Örnekler:
    - Diyalog: "Ekstrem çok yüksek geldi, bölebilir miyiz?" -> Sorgu: "Kredi kartı borcu taksitlendirme veya yeniden yapılandırma"
    - Diyalog: "Televizyon alacağım, 9 taksit oluyor mu?" -> Sorgu: "Kredi kartı mal ve hizmet alımları taksit sınırları elektronik eşya"
    - Diyalog: "150 bin çekeceğim, en fazla kaç ay olur?" -> Sorgu: "İhtiyaç kredisi vade sınırları"
    - Diyalog: "Faiziniz çok yüksek değil mi?" -> Sorgu: "Kredi kartı akdi faiz ve gecikme faizi tavan oranları"

    Diyalog:
    ---
    Müşteri: {customer_query}
    Temsilci: {agent_response}
    ---
    
    Bu diyaloğa en uygun resmi arama sorgusunu JSON formatında oluştur.
    {format_instructions}
    """
    
    prompt = ChatPromptTemplate.from_template(
        template=prompt_template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    # Zincir (prompt | llm | parser) SearchQuery nesnesi döndürür
    # Biz sadece string olan 'search_query' alanını almak için .search_query ekliyoruz
    # Not: LCEL (LangChain Expression Language) tam olarak buna izin vermeyebilir,
    # bu yüzden 'ainvoke' sonrasında .search_query yapacağız.
    return prompt | llm | parser


# =================================================================
# 4. ZİNCİR 2: UYUMLULUK ANALİZ ZİNCİRİ
# =================================================================

def create_analysis_chain():
    """
    LLM Zincir 2: Soru, Cevap ve RAG Mevzuatını alıp analiz eder.
    """
    llm = ChatOpenAI(model=LLM_MODEL, openai_api_key=OPENAI_API_KEY, temperature=0)
    
    parser = PydanticOutputParser(pydantic_object=AnalysisResult)
    
    prompt_template = """
    Sen kıdemli bir BDDK uyumluluk denetçisisin. Görevin, bir çağrı merkezi 
    görüşmesinin bir bölümünü (segment) ve ilgili BDDK mevzuatını incelemektir.

    Şunları analiz et:
    1.  **İhlal Tespiti:** Temsilcinin cevabı, ilgili mevzuata bariz bir şekilde aykırı mı? 
        (Örn: Mevzuat 'max 12 ay vade' derken temsilci '18 ay yapalım' diyorsa BU BİR İHLALDİR.)
    2.  **Eksiklik Tespiti:** Temsilci, müşterinin sorusuyla doğrudan ilgili olan mevzuattaki 
        kritik bir bilgiyi (örn: yasal faiz tavanı, vade sınırı) müşteriye 
        iletmeyi unuttu mu veya eksik bilgi verdi mi? 
        (Örn: Müşteri faiz sorarken temsilci sadece banka faizini söyleyip yasal tavanı 
        gizliyorsa BU BİR EKSİKLİKTİR.)

    ---
    İLGİLİ MEVZUAT (RAG Sonucu):
    {rag_context}
    ---
    ANALİZ EDİLECEK SEGMENT:
    Müşteri Sorusu: {customer_query}
    Temsilci Cevabı: {agent_response}
    ---
    
    Bulgularını ve önerilerini JSON formatında raporla.
    {format_instructions}
    """
    
    prompt = ChatPromptTemplate.from_template(
        template=prompt_template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    return prompt | llm | parser

# =================================================================
# 5. ORKESTRASYON (TÜM ADIMLARI BİRLEŞTİRME) - GÜNCELLENDİ
# =================================================================

# Ana zincirleri bir kez oluşturup hafızada tut
_SEGMENTATION_CHAIN = create_segmentation_chain()
_QUERY_TRANSFORM_CHAIN = create_query_transformation_chain() # YENİ
_ANALYSIS_CHAIN = create_analysis_chain()
_RETRIEVER = load_vector_store_retriever()

async def run_compliance_analysis(full_transcript: str) -> List[dict]:
    """
    Bir çağrı transkripti için tam "Çift Aşamalı RAG Analizi" akışını çalıştırır.
    (GÜNCELLENDİ: Sorgu Zenginleştirme adımı eklendi)
    """
    log.info("Akış başlatıldı: Adım 1 - Segmentasyon...")
    
    try:
        # --- ADIM 1: Transkripti Soru-Cevap segmentlerine ayır ---
        transcript_segments = await _SEGMENTATION_CHAIN.ainvoke({"transcript": full_transcript})
        all_segments = transcript_segments.segments
        
        if not all_segments:
            log.warning("Transkriptte analize uygun segment bulunamadı.")
            return []
            
        log.info(f"Adım 1 tamamlandı. {len(all_segments)} adet segment bulundu.")
        
    except Exception as e:
        log.error(f"Adım 1 (Segmentasyon) hatası: {e}")
        raise

    analysis_results_for_db = []
    
    # --- ADIM 1.5, 2 & 3: Her segment için Zenginleştirme, RAG ve Analiz ---
    for i, segment in enumerate(all_segments):
        log.info(f"Segment {i+1}/{len(all_segments)} işleniyor: '{segment.customer_query[:50]}...'")
        
        try:
            # --- ADIM 1.5: Sorgu Zenginleştirme (YENİ ADIM) ---
            log.info(f" -> Adım 1.5: Sorgu Zenginleştirme...")
            query_input = {
                "customer_query": segment.customer_query,
                "agent_response": segment.agent_response
            }
            # _QUERY_TRANSFORM_CHAIN bir SearchQuery nesnesi döndürür
            transformed_query_obj: SearchQuery = await _QUERY_TRANSFORM_CHAIN.ainvoke(query_input)
            search_query = transformed_query_obj.search_query
            log.info(f" -> RAG Sorgusu Zenginleştirildi: '{search_query}'")
            
            # --- ADIM 2: Hedefli RAG (GÜNCELLENDİ) ---
            # Zenginleştirilmiş sorgu ile RAG'ı çağır
            rag_docs = await _RETRIEVER.ainvoke(search_query)
            
            rag_context = "\n---\n".join([doc.page_content for doc in rag_docs])
            
            # --- ADIM 3: Çapraz Analiz ---
            log.info(f" -> Adım 3: Çapraz Analiz yapılıyor...")
            analysis_input = {
                "rag_context": rag_context,
                "customer_query": segment.customer_query,
                "agent_response": segment.agent_response
            }
            analysis_result: AnalysisResult = await _ANALYSIS_CHAIN.ainvoke(analysis_input)
            
            # Sonucu veritabanına eklenecek formata getir
            db_entry = {
                "segment_index": i + 1,
                "customer_query": segment.customer_query,
                "agent_response": segment.agent_response,
                "rag_context": rag_context, # Hata ayıklama için alakasız gelse bile kaydediyoruz
                "violation_detected": analysis_result.violation_detected,
                "omission_detected": analysis_result.omission_detected,
                "analysis": analysis_result.analysis,
                "suggestion": analysis_result.suggestion
            }
            analysis_results_for_db.append(db_entry)
            
        except Exception as e:
            log.error(f"Segment {i+1} işlenirken hata (Zenginleştirme, RAG veya Analiz): {e}")
            continue 

    log.info(f"Tüm akış tamamlandı. {len(analysis_results_for_db)} adet başarılı analiz sonucu.")
    return analysis_results_for_db