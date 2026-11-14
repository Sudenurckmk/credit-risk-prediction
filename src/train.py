import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

def train_model(data_path="data/processed/credit_risk_cleaned.csv"):
    """
    İşlenmiş veriyi yükler, model eğitir ve kaydeder.
    """
    df = pd.read_csv(data_path)
    
    # Bağımsız değişkenler (X) ve hedef değişken (y)
    # Kredi risk veri setinde hedef sütunumuz 'loan_status'
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']
    
    # Train - Test bölünmesi (%80 train, %20 test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Model eğitimi başlıyor (Random Forest)...")
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    
    # Değerlendirme
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\n--- Sınıflandırma Raporu ---")
    print(classification_report(y_test, y_pred))
    
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Skoru: {roc_auc:.4f}")
    
    # Modeli kaydetme
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/best_model.pkl")
    print("\nModel 'models/best_model.pkl' olarak kaydedildi.")

if __name__ == "__main__":
    train_model()