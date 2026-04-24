#https://sidra.ibge.gov.br/Tabela/5457
# Área plantada ou destinada à colheita - percentual do total geral (%) [2001 - 2024]
# Rendimento médio da produção (Quilogramas por Hectare)

import pandas as pd

df_raw = pd.read_csv(
    'eda/area_plantada.csv',
    skiprows=4,
    header=None
)

city_list = df_raw.iloc[1:643, 0].reset_index(drop=True)
planted_area = df_raw.iloc[1:643, 1:].reset_index(drop=True)
planted_area.replace(['...', '-'], [None, 0], inplace=True)
planted_area = planted_area.apply(pd.to_numeric, errors='coerce').fillna(0)

years = list(range(2019, 2025))
rows = {}

for city_idx, city in enumerate(city_list):
    for year_idx, year in enumerate(years):
        if city not in rows:
            rows[city] = 0
        rows[city] += planted_area.iloc[city_idx, year_idx]

df = pd.DataFrame(rows.items(), columns=["city", "value"])
df = df.sort_values(by="value", ascending=False)
print(df.head(25))