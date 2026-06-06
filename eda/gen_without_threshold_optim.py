import os
import pandas as pd
import re
from sklearn.metrics import classification_report

RAW_PRED_FOLDER = "results/pse_cnn/raw_predictions/"
OUTPUT_BASE = "results/pse_cnn/variations/without_threshold_optim"
OUTPUT_FOLDER = os.path.join(OUTPUT_BASE, "classification_reports")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

pattern = re.compile(r"predictions_year_(\d+).*seed_(\d+)")

if not os.path.exists(RAW_PRED_FOLDER):
    print(f"Erro: A pasta {RAW_PRED_FOLDER} não existe.")
else:
    for filename in os.listdir(RAW_PRED_FOLDER):
        
        match = pattern.search(filename)
        if not match:
            print(f"Aviso: Não foi possível identificar 'year' e 'seed' no arquivo {filename}. Ignorando.")
            continue
            
        year = match.group(1)
        seed = match.group(2)
        
        filepath = os.path.join(RAW_PRED_FOLDER, filename)
        
        df = pd.read_csv(filepath)
        
        required_cols = ['y_true', 'model_probability']
        if not all(col in df.columns for col in required_cols):
            print(f"Aviso: O arquivo {filename} não contém as colunas necessárias. Ignorando.")
            continue
            
        y_true = df['y_true']
        
        y_pred = (df['model_probability'] >= 0.5).astype(float)
        
        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        report_df = pd.DataFrame(report_dict).T
        
        out_filename = f"test_year_{year}_seed_{seed}.csv"
        out_filepath = os.path.join(OUTPUT_FOLDER, out_filename)
        
        report_df.to_csv(out_filepath, index=True)
        print(f"Relatório gerado com sucesso: {out_filepath}")

print("Processamento concluído!")
