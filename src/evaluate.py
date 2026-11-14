import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def evaluate_model(model_path="models/best_model.pkl", data_path="data/processed/credit_risk_cleaned.csv"):
    """
    Eğitilen modeli yükler ve Özellik Önem Düzeyi (Feature Importance) grafiğini çizer.
    """
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    
    X = df.drop(columns=['loan_status'])
    
    # Feature Importance (Özellik Önem Düzeyi)
    importances = model.feature_importances_
    feature_names = X.columns
    
    feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_imp = feat_imp.sort_values(by='Importance', ascending=False)
    
    print("\n--- En Önemli Özellikler ---")
    print(feat_imp.head())
    
    # Grafik oluşturma ve kaydetme
    os.makedirs("reports/figures", exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_imp.head(10))
    plt.title('Kredi Risk Modeli - Özellik Önem Düzeyi (Feature Importance)')
    plt.tight_layout()
    plt.savefig('reports/figures/feature_importance.png')
    print("\nÖzellik önem grafiği 'reports/figures/feature_importance.png' olarak kaydedildi.")

if __name__ == "__main__":
    evaluate_model()