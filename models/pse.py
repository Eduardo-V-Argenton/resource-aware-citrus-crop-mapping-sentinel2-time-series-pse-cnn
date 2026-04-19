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
# CONFIGURAÇÕES INICIAIS
# =====================================================================
FICHEIRO_INDEX = '/mnt/SSD_SATA/dataset/dataset_index.csv'
PASTA_TENSORES = "dataset/Tensors/"  
COLUNA_ALVO = "label_ia"
CANAIS_ENTRADA = 10
AMOSTRAS_S = 128  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch configurado para usar: {device}")


# -------------------------------------------------------------------------
# 1. DATALOADER DO PSE (Simplificado)
# -------------------------------------------------------------------------
class PSEDataset(Dataset):
    def __init__(self, lista_nomes_base, labels, pasta, amostras=64):
        self.lista_nomes_base = lista_nomes_base
        self.labels = labels
        self.pasta = pasta
        self.amostras = amostras

    def __len__(self):
        return len(self.lista_nomes_base)

    def __getitem__(self, idx):
        nome_base = self.lista_nomes_base[idx]
        label = self.labels[idx]

        # Carrega o tensor (Tempo, Canais, Pixels)
        tensor = np.load(os.path.join(self.pasta, f"{nome_base}_pse.npy"))
        total_pixels = tensor.shape[2]

        # Amostragem simples de pixels (com ou sem reposição)
        replace = total_pixels < self.amostras
        indices = np.random.choice(total_pixels, self.amostras, replace=replace)
        tensor_amostrado = tensor[:, :, indices]

        tensor_final = torch.from_numpy(tensor_amostrado.astype(np.float32))
        return tensor_final, torch.tensor([label], dtype=torch.float32)


# -------------------------------------------------------------------------
# 2. ARQUITETURA PIXEL-SET ENCODER + TEMPORAL
# -------------------------------------------------------------------------
class PhenologyPSE(nn.Module):
    def __init__(self, in_channels=8):
        super(PhenologyPSE, self).__init__()

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

        x = F.gelu(self.spatial_bn1(self.spatial_mlp1(x)))
        x = F.gelu(self.spatial_bn2(self.spatial_mlp2(x)))

        # Pooling Híbrido
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
            for batch_x, batch_y in loader:
                batch_probs = torch.sigmoid(model(batch_x.to(device))).cpu().numpy()
                probs_passo.extend(batch_probs)
                
                # Só precisamos guardar o gabarito na primeira passada
                if passo == 0:
                    y_true_passo.extend(batch_y.numpy())
            
            if passo == 0:
                probs_acumuladas = np.array(probs_passo)
                y_true_final = np.array(y_true_passo)
            else:
                probs_acumuladas += np.array(probs_passo)
                
    # Tira a média das 5 previsões espaciais
    probs_finais = probs_acumuladas / tta_steps
    return y_true_final.flatten(), probs_finais.flatten()


# -------------------------------------------------------------------------
# 3. O LOOP DO EXPERIMENTO
# -------------------------------------------------------------------------
df_index = pd.read_csv(FICHEIRO_INDEX)
df_index = df_index[df_index["ano"] >= 2019]

print("Verificando a integridade física dos tensores PSE...")
ficheiros_existentes = set([
    f.replace("_pse.npy", "")
    for f in os.listdir(PASTA_TENSORES) if f.endswith("_pse.npy")
])

df_index = df_index[df_index["name"].isin(ficheiros_existentes)].copy()

anos_teste = [2024, 2023, 2022]
seeds = [42, 43, 44]

BATCH_SIZE = 128
EPOCHS = 100
PATIENCE = 25 

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
            PASTA_TENSORES, amostras=AMOSTRAS_S
        )
        val_dataset = PSEDataset(
            df_val["name"].values, df_val[COLUNA_ALVO].values,
            PASTA_TENSORES, amostras=AMOSTRAS_S
        )
        test_dataset = PSEDataset(
            df_test_idx["name"].values, df_test_idx[COLUNA_ALVO].values,
            PASTA_TENSORES, amostras=AMOSTRAS_S
        )

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

        model = PhenologyPSE(in_channels=CANAIS_ENTRADA).to(device)
        model = torch.compile(model, mode="default")

        # Loss Simplificada com pos_weight
        num_negativos = len(df_train[df_train[COLUNA_ALVO] == 0])
        num_positivos = len(df_train[df_train[COLUNA_ALVO] == 1])
        peso_citrus = torch.tensor([num_negativos / num_positivos], dtype=torch.float32).to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=peso_citrus)
        
        optimizer = optim.RAdam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_model_wts = copy.deepcopy(model.state_dict())

        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0

            loop_treino = tqdm(train_loader, desc=f"Época {epoch + 1:03d}/{EPOCHS} [Treino]", leave=False)
            for batch_x, batch_y in loop_treino:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
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
                for batch_x, batch_y in loop_val:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    loss = criterion(model(batch_x), batch_y)
                    val_loss += loss.item() * batch_x.size(0)

            val_epoch_loss = val_loss / len(val_loader.dataset)
            scheduler.step()

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
        # EXTRAÇÃO DE RESULTADOS
        # =====================================================================
        print("\nCarregando os melhores pesos para gerar as métricas...")
        model.load_state_dict(best_model_wts)

        y_true_val, probs_val = prever(val_loader)
        y_true_test, probs_test = prever(test_loader)

        precisions, recalls, thresholds = precision_recall_curve(y_true_val, probs_val)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        melhor_indice = np.argmax(f1_scores[:-1]) if len(f1_scores) > len(thresholds) else np.argmax(f1_scores)
        limiar_otimo = thresholds[melhor_indice]

        print(f">> Limiar Otimizado na Validação: {limiar_otimo:.4f}")
        preds_val = (probs_val >= limiar_otimo).astype(int)
        preds_test = (probs_test >= limiar_otimo).astype(int)

        print(f">> Resultado no VAL ({ano_teste} - {seed}):")
        print(classification_report(y_true_val, preds_val, zero_division=0))
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
df_tabela.to_csv("auditoria_completa_mestrado_pse.csv")