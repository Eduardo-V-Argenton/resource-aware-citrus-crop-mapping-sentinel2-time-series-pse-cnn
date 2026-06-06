import os
import glob
import pandas as pd
from sklearn.metrics import classification_report

INDEX_FILE = '/mnt/ssd_sata/dataset/dataset_index.csv'
PREDICTIONS_FOLDER = 'results/pse_cnn/raw_predictions/'

def executar_auditoria_global(index_path, folder_path):
    prediction_files = glob.glob(os.path.join(folder_path, '*.csv'))
    
    if not prediction_files:
        print(f"ERRO: Nenhum ficheiro encontrado em {folder_path}")
        return

    print(f"A iniciar Auditoria Global em {len(prediction_files)} ficheiros (seeds/anos)...\n")
    
    df_index = pd.read_csv(index_path)
    
    lista_dfs = [pd.read_csv(f) for f in prediction_files]
    df_todas_preds = pd.concat(lista_dfs, ignore_index=True)

    df_merged = pd.merge(df_todas_preds, df_index, left_on='id_sample', right_on='name', how='inner')

    y_true = df_merged['y_true']
    y_pred = df_merged['final_threshold_prediction']

    print("="*65)
    print(" 1. RELATÓRIO DE MÉTRICAS GLOBAIS (Precision, Recall, F1)")
    print("="*65)
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=['Classe 0 (Não-Citrus)', 'Classe 1 (Citrus)'], 
        digits=4
    )
    print(report)

    print("\n" + "="*65)
    print(" 2. RAIOS-X AOS FALSOS POSITIVOS (FP) DA CLASSE 1")
    print("    (O que mais confundiu o modelo em todas as execuções?)")
    print("="*65)
    
    fp_mask = (df_merged['y_true'] == 0) & (df_merged['final_threshold_prediction'] == 1)
    df_fp = df_merged[fp_mask]

    if not df_fp.empty:
        fp_por_cultura = df_fp['crop'].value_counts().reset_index()
        fp_por_cultura.columns = ['Cultura (Classe 0)', 'Total de Falsos Alarmes']
        fp_por_cultura['Impacto nos Erros (%)'] = (fp_por_cultura['Total de Falsos Alarmes'] / len(df_fp)) * 100
        print(fp_por_cultura.to_string(index=False, float_format="%.1f"))
    else:
        print("Nenhum Falso Positivo registado.")

    print("\n" + "="*65)
    print(" 3. RAIOS-X AOS FALSOS NEGATIVOS (FN) DA CLASSE 1")
    print("    (Em que cidades a omissão de Citrus é mais grave?)")
    print("="*65)
    
    fn_mask = (df_merged['y_true'] == 1) & (df_merged['final_threshold_prediction'] == 0)
    df_fn = df_merged[fn_mask]

    if not df_fn.empty:
        fn_por_cidade = df_fn['city'].value_counts().reset_index()
        fn_por_cidade.columns = ['Cidade com Omissão', 'Total de Omissões']
        fn_por_cidade['Impacto nos Erros (%)'] = (fn_por_cidade['Total de Omissões'] / len(df_fn)) * 100
        print(fn_por_cidade.head(10).to_string(index=False, float_format="%.1f"))
    else:
        print("Nenhum Falso Negativo registado.")

if __name__ == "__main__":
    executar_auditoria_global(INDEX_FILE, PREDICTIONS_FOLDER)