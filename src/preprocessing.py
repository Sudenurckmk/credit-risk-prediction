import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def handle_missing_values(df):
    """
    Eksik verileri uygun stratejilerle doldurur.
    - loan_int_rate: medyan ile doldurulur (aykırı değerlerden etkilenmemesi için).
    - person_emp_length: medyan veya ortalama ile doldurulur.
    """
    df['loan_int_rate'].fillna(df['loan_int_rate'].median(), inplace=True)
    df['person_emp_length'].fillna(df['person_emp_length'].median(), inplace=True)
    print("Eksik veriler başarıyla dolduruldu.")
    return df

def feature_engineering(df):
    """
    Ham veriden yeni ve anlamlı özellikler türetir.
    - Gelir ve kredi tutarına bağlı oranlar / gruplar oluşturulabilir.
    """
    # Örnek: Kredi tutarının yıllık gelire oranı zaten var (loan_percent_income),
    # Ek olarak borçlanma oranı veya yaş grupları türetebiliriz.
    df['income_to_age_ratio'] = df['person_income'] / (df['person_age'] + 1)
    print("Özellik mühendisliği tamamlandı, yeni özellikler eklendi.")
    return df

def encode_categorical_features(df):
    """
    Kategorik (metinsel) sütunları sayısal formata dönüştürür.
    """
    cat_cols = df.select_dtypes(include=['object']).columns
    
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        
    print(f"Kategorik sütunlar dönüştürüldü: list({cat_cols})")
    return df

def preprocess_pipeline(file_path="data/raw/credit_risk_dataset.csv"):
    """
    Tüm ön işleme adımlarını sırayla çalıştıran ana pipeline fonksiyonu.
    """
    df = pd.read_csv(file_path)
    df = handle_missing_values(df)
    df = feature_engineering(df)
    df = encode_categorical_features(df)
    
    # İşlenmiş veriyi kaydedebileceğimiz bir klasör kontrolü
    import os
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/credit_risk_cleaned.csv", index=False)
    print("İşlenmiş veri 'data/processed/credit_risk_cleaned.csv' olarak kaydedildi.")
    return df

if __name__ == "__main__":
    preprocess_pipeline()