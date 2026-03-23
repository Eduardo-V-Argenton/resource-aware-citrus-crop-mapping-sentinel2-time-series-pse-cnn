import csv
import os

ficheiro = 'dataset/dataset_index.csv'
ficheiro_temp = 'dataset/dataset_index_limpo.csv'

# O cabeçalho perfeito de 8 colunas que nós queremos
cabecalho_ideal = ['nome_base', 'label_ia', 'classe_mapbiomas', 'cultura_real', 'cidade', 'ano', 'area_ha', 'e_pedaco']
linhas_limpas = [cabecalho_ideal]

print("-> A iniciar a cirurgia no CSV...")

with open(ficheiro, 'r', encoding='utf-8') as f:
    leitor = csv.reader(f)
    
    for i, linha in enumerate(leitor):
        # Pula qualquer linha que seja um cabeçalho (seja o antigo de 7 ou o novo de 8)
        if 'nome_base' in linha:
            continue
            
        # Se a linha tem dados antigos (7 colunas), adicionamos um '0' no final
        if len(linha) == 7:
            linha.append('0')
            
        # Guarda a linha se ela tiver exatamente 8 colunas agora
        if len(linha) == 8:
            linhas_limpas.append(linha)
        else:
            print(f"Linha {i} ignorada por estar corrompida: {linha}")

# Substitui o ficheiro antigo pelo novo ficheiro perfeito
with open(ficheiro_temp, 'w', encoding='utf-8', newline='') as f:
    escritor = csv.writer(f)
    escritor.writerows(linhas_limpas)

# Troca os ficheiros no sistema (Arch Linux)
os.replace(ficheiro_temp, ficheiro)

print(f"-> Sucesso! O CSV foi padronizado. Total de fazendas válidas: {len(linhas_limpas) - 1}")