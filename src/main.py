from data_loader import load_data
from preprocessing import preprocess_pipeline
from train import train_model
from evaluate import evaluate_model

if __name__ == "__main__":
    print("=== KREDİ RİSK ANALİZİ PROJESİ BAŞLATILIYOR ===")
    
    # 1. Aşama: Veri Yükleme
    print("\n[1/4] Veri yükleniyor...")
    load_data()
    
    # 2. Aşama: Veri Ön İşleme ve Temizleme
    print("\n[2/4] Veri ön işleme ve temizleme yapılıyor...")
    preprocess_pipeline()
    
    # 3. Aşama: Model Eğitimi
    print("\n[3/4] Model eğitiliyor...")
    train_model()
    
    # 4. Aşama: Model Değerlendirme ve Yorumlama
    print("\n[4/4] Model değerlendiriliyor ve grafikler oluşturuluyor...")
    evaluate_model()
    
    print("\n=== PROJE BAŞARIYLA TAMAMLANDI ===")