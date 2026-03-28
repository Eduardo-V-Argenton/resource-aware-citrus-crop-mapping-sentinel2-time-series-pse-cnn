import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report

print("Carregando os datasets...")
df_features = pd.read_csv('dataset/dataset_curvas.csv')
df_original = pd.read_csv('dataset/dataset_index.csv')

COLUNA_ALVO = 'label_ia' 

# Precisamos trazer a coluna 'ano' do CSV original para fazer a separação temporal
df_completo = pd.merge(df_features, df_original[['nome_base', COLUNA_ALVO, 'ano']], on='nome_base', how='inner')

# Configurações do Experimento
anos_teste = [2024, 2023, 2022]
seeds = [42, 43, 44]

for ano_teste in anos_teste:
    print("\n" + "#"*80)
    print(f" EXPERIMENTO: TESTE DE INDEPENDÊNCIA TEMPORAL - ANO {ano_teste} ")
    print("#"*80)
    
    # 1. Separa o ano escolhido EXCLUSIVAMENTE para Teste (modelo nunca vai ver isso no treino)
    df_test = df_completo[df_completo['ano'] == ano_teste]
    
    # 2. O restante dos anos será usado para Treino e Validação
    df_resto = df_completo[df_completo['ano'] != ano_teste]
    
    # Isolando X e y do Teste
    X_test = df_test.drop(columns=['nome_base', COLUNA_ALVO, 'ano'])
    y_test = df_test[COLUNA_ALVO]
    
    # Isolando X e y do Resto (Treino + Validação)
    X_resto = df_resto.drop(columns=['nome_base', COLUNA_ALVO, 'ano'])
    y_resto = df_resto[COLUNA_ALVO]
    
    # Rodando as 3 sementes aleatórias para o ano atual
    for seed in seeds:
        print(f"\n" + "-"*50)
        print(f" RUN: Seed (Semente) {seed}")
        print("-"*50)
        
        # 3. Divisão de Treino (aprox 70%) e Validação (aprox 15%)
        # A proporção de 0.1765 do "resto" simula os 15% do total do dataset
        X_train, X_val, y_train, y_val = train_test_split(
            X_resto, y_resto, 
            test_size=0.1765, 
            random_state=seed, 
            stratify=y_resto
        )
        
        print(f"Tamanhos -> Treino: {len(X_train)} | Validação: {len(X_val)} | Teste (Ano {ano_teste}): {len(X_test)}")
        proporcao_negativos = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        # 4. Treinamento
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,          # Número de árvores
            learning_rate=0.05,        # Taxa de aprendizado baixa (aprende devagar e com segurança)
            max_depth=5,               # Árvores rasas (essencial para não decorar a série temporal)
            subsample=0.8,             # Usa apenas 80% dos talhões a cada árvore
            colsample_bytree=0.8,      # Usa apenas 80% das features (dias/índices) a cada árvore
            reg_alpha=0.1,             # Regularização L1 (Penaliza modelos complexos)
            reg_lambda=1.0,            # Regularização L2 (Evita pesos extremos)
            scale_pos_weight=proporcao_negativos, 
            random_state=seed,
            n_jobs=-1,
            eval_metric='logloss'      # Métrica interna de avaliação
        )
        xgb_model.fit(X_train, y_train)
        
        # 5. Previsões
        y_pred_val = xgb_model.predict(X_val)
        y_pred_test = xgb_model.predict(X_test)
        
        # 6. Relatórios
        print(f"\n>> Desempenho na VALIDAÇÃO (Anos Misturados):")
        print(classification_report(y_val, y_pred_val, zero_division=0))
        
        print(f">> Desempenho no TESTE (Ano Inédito {ano_teste}):")
        print(classification_report(y_test, y_pred_test, zero_division=0))
        
        # Mostra as features mais importantes apenas na primeira seed para não poluir muito a tela
        if seed == 42:
            importancias = xgb_model.feature_importances_
            df_importancia = pd.DataFrame({'Feature': X_train.columns, 'Importancia': importancias})
            df_importancia = df_importancia.sort_values(by='Importancia', ascending=False)
            print(" TOP 5 Variáveis Mais Importantes nesta rodada:")
            print(df_importancia.head(5).to_string(index=False))