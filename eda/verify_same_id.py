import pandas as pd

df = pd.read_csv("dataset/dataset_index.csv")

# conta quantas vezes cada id aparece
counts = df["id_poligono"].value_counts()

total = len(counts)

for i in range(1, 7):
    perc = (counts > i).sum() / total * 100
    print(f"> {i}: {perc:.2f}%")
