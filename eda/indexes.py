import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt

# =====================================================================
# 1. CONFIGURAÇÕES INICIAIS
# =====================================================================
ARQUIVO_EDA = "dataset_eda_temporal_20pct.csv"

LISTA_INDICES_INICIAIS = [
    "EVI",
    "NDRE",
    "NDTI",
    "NDMI",
    "BSI",
    "TCARI",
    "NDVI",
    "GNDVI",
    "SAVI",
    "MSI",
    "VARI",
    "MCARI",
    "CIRED_EDGE",
    "PSRI",
    "NDWI",
    "MSAVI",
    "OSAVI",
    "ARVI",
    "SIPI",
    "NDGI",
    "CI_GREEN",
]

print(f"Lendo o dataset: {ARQUIVO_EDA}...")
df = pd.read_csv(ARQUIVO_EDA)
y = df["label_ia"].values

# =====================================================================
# 2. O ALGORITMO RFE AGRUPADO (Grouped Recursive Feature Elimination)
# =====================================================================
print("\n" + "=" * 60)
print(" INICIANDO ELIMINAÇÃO RECURSIVA AGRUPADA (GROUPED RFE)")
print("=" * 60)

indices_ativos = LISTA_INDICES_INICIAIS.copy()
historico_eliminacao = []
historico_performance = []

# Modelo base para a avaliação
rf = RandomForestClassifier(
    n_estimators=150, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1
)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Loop até sobrar apenas 1 índice
while len(indices_ativos) > 0:
    print(f"\n[Avaliando modelo com {len(indices_ativos)} índices...]")

    # 1. Monta o X apenas com os índices que ainda estão "vivos"
    colunas_ativas = [f"{ind}_d{dia}" for ind in indices_ativos for dia in range(1, 26)]
    X_atual = df[colunas_ativas].values

    # 2. Mede a performance real deste conjunto (Usando F1-Score Macro para lidar com desbalanceamento)
    scores = cross_val_score(rf, X_atual, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    score_medio = np.mean(scores)
    historico_performance.append((len(indices_ativos), score_medio))

    print(f" -> F1-Score Macro: {score_medio:.4f}")

    # Se só sobrou 1 índice, registramos ele e paramos o loop
    if len(indices_ativos) == 1:
        historico_eliminacao.append(
            {"Indice": indices_ativos[0], "Motivo": "Último Sobrevivente (Mais Forte)"}
        )
        break

    # 3. Treina o RF completo para ver quem é o pior índice do grupo atual
    rf.fit(X_atual, y)
    importancias_brutas = rf.feature_importances_

    # 4. Soma a importância dos 25 dias para descobrir a "Nota Global" de cada índice
    notas_indices = {}
    inicio = 0
    for ind in indices_ativos:
        nota_global = np.sum(importancias_brutas[inicio : inicio + 25])
        notas_indices[ind] = nota_global
        inicio += 25

    # 5. Descobre quem foi o pior índice e o elimina!
    pior_indice = min(notas_indices, key=notas_indices.get)
    pior_nota = notas_indices[pior_indice]

    print(
        f" -> Índice mais fraco detectado: {pior_indice} (Importância: {pior_nota * 100:.2f}%). ELIMINADO!"
    )

    historico_eliminacao.append(
        {
            "Indice": pior_indice,
            "Motivo": f"Eliminado quando restavam {len(indices_ativos)}",
        }
    )
    indices_ativos.remove(pior_indice)

# =====================================================================
# 3. EXIBIÇÃO DE RESULTADOS E PLOTAGEM
# =====================================================================
print("\n" + "=" * 60)
print(" RANKING FINAL DE SOBREVIVÊNCIA (Do Pior para o Melhor)")
print("=" * 60)

# O ranking final é o inverso da ordem de eliminação
ranking_final = [item["Indice"] for item in historico_eliminacao][::-1]

for i, ind in enumerate(ranking_final):
    print(f"{i + 1}º Lugar: {ind}")

# ---------------------------------------------------------
# Plotando a Curva de Performance do RFE
# ---------------------------------------------------------
qtd_indices, scores_f1 = zip(*historico_performance)

# Inverte as listas para o gráfico ir da esquerda (1 índice) para a direita (21 índices)
qtd_indices = list(qtd_indices)[::-1]
scores_f1 = list(scores_f1)[::-1]

plt.figure(figsize=(10, 6))
plt.plot(
    qtd_indices,
    scores_f1,
    marker="o",
    linestyle="-",
    color="b",
    linewidth=2,
    markersize=8,
)
plt.title("RFE: Desempenho vs Número de Índices Fenológicos", fontsize=14)
plt.xlabel("Quantidade de Índices Utilizados", fontsize=12)
plt.ylabel("F1-Score Macro (Cross-Validation)", fontsize=12)
plt.xticks(qtd_indices)
plt.grid(True, linestyle="--", alpha=0.7)

# Destaca o ponto máximo (O "Sweet Spot")
melhor_idx = np.argmax(scores_f1)
plt.scatter(
    qtd_indices[melhor_idx],
    scores_f1[melhor_idx],
    color="red",
    s=200,
    zorder=5,
    label=f"Pico Máximo ({qtd_indices[melhor_idx]} Índices)",
)
plt.legend()

plt.tight_layout()
plt.savefig("grafico_rfe_agrupado.png", dpi=300)
print("\n[OK] Gráfico salvo como 'grafico_rfe_agrupado.png'!")
print(
    f"-> Dica: Olhe o gráfico. O pico vermelho mostra EXATAMENTE quantos índices do topo do ranking você deve usar na CNN!"
)
