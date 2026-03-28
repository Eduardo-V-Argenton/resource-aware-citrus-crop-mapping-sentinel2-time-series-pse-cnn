import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
        
        # 4. Treinamento
        rf_model = RandomForestClassifier(
            n_estimators=300,          # Mais árvores para compensar a poda
            max_depth=12,              # Impede a árvore de crescer infinitamente e decorar os dados
            min_samples_leaf=5,        # Regras finais devem valer para pelo menos 5 talhões
            max_features='sqrt',       # Sorteia poucas features por nó para diversificar a floresta
            random_state=seed, 
            n_jobs=-1, 
            class_weight='balanced'
        )
        rf_model.fit(X_train, y_train)
        
        # 5. Previsões
        y_pred_val = rf_model.predict(X_val)
        y_pred_test = rf_model.predict(X_test)
        
        # 6. Relatórios
        print(f"\n>> Desempenho na VALIDAÇÃO (Anos Misturados):")
        print(classification_report(y_val, y_pred_val, zero_division=0))
        
        print(f">> Desempenho no TESTE (Ano Inédito {ano_teste}):")
        print(classification_report(y_test, y_pred_test, zero_division=0))
        
        # Mostra as features mais importantes apenas na primeira seed para não poluir muito a tela
        if seed == 42:
            importancias = rf_model.feature_importances_
            df_importancia = pd.DataFrame({'Feature': X_train.columns, 'Importancia': importancias})
            df_importancia = df_importancia.sort_values(by='Importancia', ascending=False)
            print(" TOP 5 Variáveis Mais Importantes nesta rodada:")
            print(df_importancia.head(5).to_string(index=False))