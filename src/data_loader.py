import pandas as pd
import os

def load_data(file_path="data/raw/credit_risk_dataset.csv"):
    """
    Kredi risk veri setini belirtilen yoldan yükler.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} konumunda veri seti bulunamadı!")
    
    df = pd.read_csv(file_path)
    print(f"Veri seti başarıyla yüklendi. Satır sayısı: {df.shape[0]}, Sütun sayısı: {df.shape[1]}")
    return df

if __name__ == "__main__":
    load_data()