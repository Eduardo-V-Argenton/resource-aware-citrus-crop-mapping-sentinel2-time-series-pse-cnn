import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler 

print("Carregando os datasets...")
df_features = pd.read_csv('dataset/dataset_curvas.csv')
df_original = pd.read_csv('dataset/dataset_index.csv')

COLUNA_ALVO = 'label_ia' 

df_completo = pd.merge(df_features, df_original[['nome_base', COLUNA_ALVO, 'ano']], on='nome_base', how='inner')

anos_teste = [2024, 2023, 2022]
seeds = [42, 43, 44]

for ano_teste in anos_teste:
    print("\n" + "#"*80)
    print(f" EXPERIMENTO: SVM (Support Vector Machine) - ANO TESTE {ano_teste} ")
    print("#"*80)
    
    df_test = df_completo[df_completo['ano'] == ano_teste]
    df_resto = df_completo[df_completo['ano'] != ano_teste]
    
    X_test = df_test.drop(columns=['nome_base', COLUNA_ALVO, 'ano'])
    y_test = df_test[COLUNA_ALVO]
    
    X_resto = df_resto.drop(columns=['nome_base', COLUNA_ALVO, 'ano'])
    y_resto = df_resto[COLUNA_ALVO]
    
    for seed in seeds:
        print(f"\n" + "-"*50)
        print(f" RUN: Seed {seed}")
        print("-"*50)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_resto, y_resto, 
            test_size=0.1765, 
            random_state=seed, 
            stratify=y_resto
        )
        
        # -------------------------------------------------------------------------
        # NOVO PASSO VITAL: ESCALONAMENTO DOS DADOS (Standardization)
        # -------------------------------------------------------------------------
        scaler = StandardScaler()
        
        # O modelo "aprende" a escala apenas com os dados de Treino (evita vazamento de dados)
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Aplica a mesma régua de escala para Validação e Teste
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # -------------------------------------------------------------------------
        # CONFIGURAÇÃO DO SVM
        # -------------------------------------------------------------------------
        svm_model = SVC(
            kernel='rbf',               # Radial Basis Function: lida excelente com dados não-lineares
            C=1.0,                      # Parâmetro de Regularização (menor = mais generalização)
            class_weight='balanced',    # Lida com o desbalanceamento entre laranja e outras culturas
            random_state=seed
        )
        
        # Treinamento (Atenção: o SVM treina usando a matriz escalonada!)
        svm_model.fit(X_train_scaled, y_train)
        
        # Previsões
        y_pred_val = svm_model.predict(X_val_scaled)
        y_pred_test = svm_model.predict(X_test_scaled)
        
        # Relatórios
        print(f"\n>> Desempenho na VALIDAÇÃO (Anos Misturados):")
        print(classification_report(y_val, y_pred_val, zero_division=0))
        
        print(f">> Desempenho no TESTE (Ano Inédito {ano_teste}):")
        print(classification_report(y_test, y_pred_test, zero_division=0))