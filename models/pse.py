import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

torch.set_float32_matmul_precision('high')

# =====================================================================
# CONFIGURAÇÕES INICIAIS E CRIAÇÃO DE DIRETÓRIOS PARA O ARTIGO
# =====================================================================
FICHEIRO_INDEX = '/mnt/SSD_SATA/dataset/dataset_index.csv'
PASTA_TENSORES = "dataset/Tensors/"  
COLUNA_ALVO = "label_ia"
CANAIS_ENTRADA = 10
AMOSTRAS_S = 128  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch configurado para usar: {device}")

# Criando estrutura de diretórios para salvar os artefatos do artigo
DIR_BASE_RESULTADOS = "resultados_artigo"
os.makedirs(DIR_BASE_RESULTADOS, exist_ok=True)
os.makedirs(os.path.join(DIR_BASE_RESULTADOS, "modelos"), exist_ok=True)
os.makedirs(os.path.join(DIR_BASE_RESULTADOS, "historico_loss"), exist_ok=True)
os.makedirs(os.path.join(DIR_BASE_RESULTADOS, "relatorios_classificacao"), exist_ok=True)
os.makedirs(os.path.join(DIR_BASE_RESULTADOS, "previsoes_brutas"), exist_ok=True)
os.makedirs(os.path.join(DIR_BASE_RESULTADOS, "rastreio_lotes"), exist_ok=True)

# -------------------------------------------------------------------------
# 1. DATALOADER DO PSE
# -------------------------------------------------------------------------
class PSEDataset(Dataset):
    def __init__(self, lista_nomes_base, labels, pasta, amostras=64, is_train=False):
        self.lista_nomes_base = lista_nomes_base
        self.labels = labels
        self.pasta = pasta
        self.amostras = amostras
        self.is_train = is_train

    def __len__(self):
        return len(self.lista_nomes_base)

    def __getitem__(self, idx):
        nome_base = self.lista_nomes_base[idx]
        label = self.labels[idx]

        # Carrega o tensor (Tempo, Canais, Pixels)
        tensor = np.load(os.path.join(self.pasta, f"{nome_base}_pse.npy"))
        total_pixels = tensor.shape[2]

        replace = total_pixels < self.amostras
        indices = np.random.choice(total_pixels, self.amostras, replace=replace)
        tensor_amostrado = tensor[:, :, indices]
        
        if self.is_train and np.random.rand() > 0.5:
            t_idx = np.random.randint(0, tensor_amostrado.shape[0])
            tensor_amostrado[t_idx, :, :] = 0.0
            
        tensor_final = torch.from_numpy(tensor_amostrado.astype(np.float32))
        
        return tensor_final, torch.tensor([label], dtype=torch.float32), nome_base


# -------------------------------------------------------------------------
# 2. ARQUITETURA PIXEL-SET ENCODER + TEMPORAL
# -------------------------------------------------------------------------
class PhenologyPSE(nn.Module):
    def __init__(self, in_channels=10): 
        super(PhenologyPSE, self).__init__()

        self.input_norm = nn.BatchNorm1d(in_channels) 
        
        self.spatial_mlp1 = nn.Conv1d(in_channels, 32, kernel_size=1)
        self.spatial_bn1 = nn.BatchNorm1d(32)

        self.spatial_mlp2 = nn.Conv1d(32, 64, kernel_size=1)
        self.spatial_bn2 = nn.BatchNorm1d(64)

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        B, T, C, S = x.size()
        
        x = x.view(B * T, C, S)
        
        x = self.input_norm(x) 
                 
        x = F.gelu(self.spatial_bn1(self.spatial_mlp1(x)))
        x = F.gelu(self.spatial_bn2(self.spatial_mlp2(x)))

        x_max = torch.max(x, dim=2)[0]
        x_mean = torch.mean(x, dim=2)
        x_pooled = torch.cat([x_max, x_mean], dim=1) 

        x_temp = x_pooled.view(B, T, -1).permute(0, 2, 1)
        
        x_temp = self.temporal_conv(x_temp) 
        
        x_flat = x_temp.view(B, -1)
        out = self.classifier(x_flat)

        return out


def prever(loader, tta_steps=5):
    """Função de predição com Test-Time Augmentation (TTA)"""
    model.eval()
    y_true_final = []
    probs_acumuladas = None
    
    with torch.no_grad():
        for passo in range(tta_steps):
            y_true_passo, probs_passo = [], []
            for batch_x, batch_y, _ in loader:
                batch_probs = torch.sigmoid(model(batch_x.to(device))).cpu().numpy()
                probs_passo.extend(batch_probs)
                
                if passo == 0:
                    y_true_passo.extend(batch_y.numpy())
            
            if passo == 0:
                probs_acumuladas = np.array(probs_passo)
                y_true_final = np.array(y_true_passo)
            else:
                probs_acumuladas += np.array(probs_passo)
                
    probs_finais = probs_acumuladas / tta_steps
    return y_true_final.flatten(), probs_finais.flatten()


# -------------------------------------------------------------------------
# 3. O LOOP DO EXPERIMENTO
# -------------------------------------------------------------------------
df_index = pd.read_csv(FICHEIRO_INDEX)

print("Verificando a integridade física dos tensores PSE...")
ficheiros_existentes = set([
    f.replace("_pse.npy", "")
    for f in os.listdir(PASTA_TENSORES) if f.endswith("_pse.npy")
])

df_index = df_index[df_index["name"].isin(ficheiros_existentes)].copy()

anos_teste = [2023, 2022]
seeds = [42, 43, 44]

BATCH_SIZE = 128
EPOCHS = 100
PATIENCE = 15 

resultados_gerais = {}

for ano_teste in anos_teste:
    print(f"\n{'=' * 80}\n TESTE INÉDITO: ANO {ano_teste} \n{'=' * 80}")

    df_test_idx = df_index[df_index["ano"] == ano_teste].copy()
    df_resto_idx = df_index[df_index["ano"] != ano_teste].copy()

    for seed in seeds:
        print(f"\n{'-' * 50}\n RODADA: Seed {seed}\n{'-' * 50}")

        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        gerador_divisao = sgkf.split(
            df_resto_idx,
            df_resto_idx["mapbiomas_class"],
            groups=df_resto_idx["id"],
        )
        train_indices, val_indices = next(gerador_divisao)

        df_val = df_resto_idx.iloc[val_indices].copy()
        df_train_bruto = df_resto_idx.iloc[train_indices].copy()
        df_train = df_train_bruto.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        train_dataset = PSEDataset(
            df_train["name"].values, df_train[COLUNA_ALVO].values,
            PASTA_TENSORES, amostras=AMOSTRAS_S, is_train=True
        )
        val_dataset = PSEDataset(
            df_val["name"].values, df_val[COLUNA_ALVO].values,
            PASTA_TENSORES, amostras=AMOSTRAS_S
        )
        test_dataset = PSEDataset(
            df_test_idx["name"].values, df_test_idx[COLUNA_ALVO].values,
            PASTA_TENSORES, amostras=AMOSTRAS_S
        )

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

        model = PhenologyPSE(in_channels=CANAIS_ENTRADA).to(device)

        num_negativos = len(df_train[df_train[COLUNA_ALVO] == 0])
        num_positivos = len(df_train[df_train[COLUNA_ALVO] == 1])
        peso_citrus = torch.tensor([num_negativos / num_positivos], dtype=torch.float32).to(device)

        criterion = nn.BCEWithLogitsLoss(reduction='none').to(device)
        
        optimizer = optim.RAdam(model.parameters(), lr=1e-4, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_model_wts = copy.deepcopy(model.state_dict())
        
        # Dicionário para rastrear a evolução do modelo para geração de gráficos
        historico_treino = {"epoca": [], "train_loss": [], "val_loss": []}
        
        rastreio_lotes = []
        
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0

            loop_treino = tqdm(train_loader, desc=f"Época {epoch + 1:03d}/{EPOCHS} [Treino]", leave=False)
            for step, (batch_x, batch_y, batch_nomes) in enumerate(loop_treino):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                rastreio_lotes.append({
                    "epoca": epoch + 1,
                    "id_lote": step,
                    "fase": "train",
                    "amostras": "|".join(batch_nomes)
                })
                
                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                
                running_loss += loss.item() * batch_x.size(0)
                loop_treino.set_postfix(loss=loss.item())

            epoch_loss = running_loss / len(train_loader.dataset)

            model.eval()
            val_loss = 0.0
            loop_val = tqdm(val_loader, desc=f"Época {epoch + 1:03d}/{EPOCHS} [Valida]", leave=False)
            
            with torch.no_grad():
                for step, (batch_x, batch_y, batch_nomes) in enumerate(loop_val):
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    rastreio_lotes.append({
                        "epoca": epoch + 1,
                        "fase": "val",
                        "id_lote": step,
                        "amostras": "|".join(batch_nomes)
                    })
                    loss = criterion(model(batch_x), batch_y)
                    val_loss += loss.item() * batch_x.size(0)

            val_epoch_loss = val_loss / len(val_loader.dataset)
            scheduler.step()
            
            # Salvando os dados no histórico
            historico_treino["epoca"].append(epoch + 1)
            historico_treino["train_loss"].append(epoch_loss)
            historico_treino["val_loss"].append(val_epoch_loss)

            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
                status = " (Melhorou!)"
            else:
                epochs_no_improve += 1
                status = ""

            print(f"Época {epoch + 1:03d}/{EPOCHS} | Train: {epoch_loss:.4f} | Val: {val_epoch_loss:.4f}{status}")

            if epochs_no_improve >= PATIENCE:
                print(f"-> Early Stopping: O treinamento parou na época {epoch + 1}.")
                break

        # =====================================================================
        # EXPORTAÇÃO DOS ARTEFATOS E RESULTADOS (NOVO)
        # =====================================================================
        print("\nCarregando os melhores pesos para gerar as métricas e exportar dados...")
        model.load_state_dict(best_model_wts)
        
        # 1. Salvar os pesos do modelo treinado
        caminho_modelo = os.path.join(DIR_BASE_RESULTADOS, "modelos", f"pesos_ano_{ano_teste}_seed_{seed}.pth")
        torch.save(best_model_wts, caminho_modelo)
        
        # 2. Salvar o histórico de aprendizado (Loss)
        df_historico = pd.DataFrame(historico_treino)
        df_historico.to_csv(os.path.join(DIR_BASE_RESULTADOS, "historico_loss", f"loss_ano_{ano_teste}_seed_{seed}.csv"), index=False)

        y_true_val, probs_val = prever(val_loader)
        y_true_test, probs_test = prever(test_loader)

        precisions, recalls, thresholds = precision_recall_curve(y_true_val, probs_val)
        recalls = recalls[:-1]
        precisions = precisions[:-1]
        mask = recalls >= 0.75
        if np.any(mask):
            f1_scores = 2 * (precisions[mask] * recalls[mask]) / (precisions[mask] + recalls[mask] + 1e-8)
            melhor_indice = np.argmax(f1_scores)
            limiar_otimo = thresholds[mask][melhor_indice]
        else:
            melhor_indice = np.argmax(recalls)
            limiar_otimo = thresholds[melhor_indice]

        print(f">> Limiar Otimizado na Validação: {limiar_otimo:.4f}")
        preds_val = (probs_val >= limiar_otimo).astype(int)
        preds_test = (probs_test >= limiar_otimo).astype(int)

        # 3. Extrair e salvar os relatórios do scikit-learn como CSV
        report_val_dict = classification_report(y_true_val, preds_val, zero_division=0, output_dict=True)
        report_test_dict = classification_report(y_true_test, preds_test, zero_division=0, output_dict=True)
        
        pd.DataFrame(report_val_dict).transpose().to_csv(
            os.path.join(DIR_BASE_RESULTADOS, "relatorios_classificacao", f"val_ano_{ano_teste}_seed_{seed}.csv")
        )
        pd.DataFrame(report_test_dict).transpose().to_csv(
            os.path.join(DIR_BASE_RESULTADOS, "relatorios_classificacao", f"test_ano_{ano_teste}_seed_{seed}.csv")
        )

        # 4. Salvar as previsões brutas (Gabarito vs Probabilidade Real vs Chute Final)
        df_preds_brutas = pd.DataFrame({
            "y_true": y_true_test,
            "probabilidade_modelo": probs_test,
            "predicao_final_limiar": preds_test
        })
        df_preds_brutas.to_csv(
            os.path.join(DIR_BASE_RESULTADOS, "previsoes_brutas", f"previsoes_ano_{ano_teste}_seed_{seed}.csv"), 
            index=False
        )
        df_lotes = pd.DataFrame(rastreio_lotes)
        df_lotes.to_csv(
            os.path.join(DIR_BASE_RESULTADOS, "rastreio_lotes", f"composicao_lotes_ano_{ano_teste}_seed_{seed}.csv"), 
            index=False
        )

        print(f">> Resultado no TESTE ({ano_teste} - {seed}):")
        print(classification_report(y_true_test, preds_test, zero_division=0))

        df_test_current = df_test_idx.copy()
        df_test_current["previsao_IA"] = preds_test

        col_key = (ano_teste, seed)
        resultados_gerais[col_key] = {}

        for cultura, dados_cultura in df_test_current.groupby("crop"):
            total = len(dados_cultura)
            alvo_real = 1 if cultura == "Citrus" else 0
            acertos = (dados_cultura["previsao_IA"] == alvo_real).sum()
            taxa_acerto = (acertos / total) * 100
            resultados_gerais[col_key][cultura] = f"{acertos}/{total} ({taxa_acerto:.1f}%)"

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_test, preds_test, labels=[0, 1], zero_division=0
        )

        resultados_gerais[col_key]["Recall 0"] = round(recall[0], 2)
        resultados_gerais[col_key]["Recall 1"] = round(recall[1], 2)
        resultados_gerais[col_key]["Precision 0"] = round(precision[0], 2)
        resultados_gerais[col_key]["Precision 1"] = round(precision[1], 2)

# =====================================================================
# GERAÇÃO DA TABELA FINAL COMPILADA
# =====================================================================
print("\n" + "=" * 80)
print(" TABELA FINAL DE RESULTADOS (ANÁLISE FINA E MÉTRICAS GERAIS)")
print("=" * 80)

df_tabela = pd.DataFrame(resultados_gerais)
df_tabela = df_tabela[sorted(df_tabela.columns)]

culturas_ordenadas = sorted([c for c in df_index["cultura_real"].dropna().unique()])
ordem_linhas = culturas_ordenadas + ["Recall 0", "Recall 1", "Precision 0", "Precision 1"]

df_tabela = df_tabela.reindex(ordem_linhas)
df_tabela.columns.names = ["Ano", "Seed"]

print(df_tabela.to_string())
caminho_tabela_final = os.path.join(DIR_BASE_RESULTADOS, "tabela_consolidada_artigo.csv")
df_tabela.to_csv(caminho_tabela_final)
print(f"\nTodos os artefatos foram salvos com sucesso na pasta: {DIR_BASE_RESULTADOS}/")
