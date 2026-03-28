import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

# Para uma barra de progresso bonita no terminal
try:
    from tqdm import tqdm
except ImportError:
    print("Instale o tqdm com: pip install tqdm")
    exit()

class DatasetDualBranch(Dataset):
    def __init__(self, df, pasta_tensores, size_10m=(64, 64), size_20m=(32, 32), max_quinzenas=24):
        self.df = df.reset_index(drop=True)
        self.pasta_tensores = pasta_tensores
        self.size_10m = size_10m
        self.size_20m = size_20m
        self.max_quinzenas = max_quinzenas

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        linha = self.df.iloc[idx]
        nome_base = linha['nome_base']
        label = linha['label_ia']
        
        t_10m = np.load(os.path.join(self.pasta_tensores, f"{nome_base}_10m.npy"))
        t_20m = np.load(os.path.join(self.pasta_tensores, f"{nome_base}_20m.npy"))
        
        # Garante o tempo estático (24 quinzenas)
        t_10m = self._pad_ou_cortar_tempo(t_10m)
        t_20m = self._pad_ou_cortar_tempo(t_20m)
        
        t10 = torch.tensor(t_10m, dtype=torch.float32)
        t20 = torch.tensor(t_20m, dtype=torch.float32)
        
        # Spatial Resize
        t10 = F.interpolate(t10, size=self.size_10m, mode='bilinear', align_corners=False)
        t20 = F.interpolate(t20, size=self.size_20m, mode='bilinear', align_corners=False)
        
        # Early Time Fusion (Esmaga o tempo nos canais)
        # 10m: (24, 4, 64, 64) -> (96, 64, 64)
        img_10m_flat = t10.view(-1, self.size_10m[0], self.size_10m[1])
        # 20m: (24, 6, 32, 32) -> (144, 32, 32)
        img_20m_flat = t20.view(-1, self.size_20m[0], self.size_20m[1])
        
        return img_10m_flat, img_20m_flat, torch.tensor(label, dtype=torch.long)

    def _pad_ou_cortar_tempo(self, tensor):
        t_atual = tensor.shape[0]
        if t_atual > self.max_quinzenas:
            return tensor[:self.max_quinzenas, ...]
        elif t_atual < self.max_quinzenas:
            pad_shape = (self.max_quinzenas - t_atual, tensor.shape[1], tensor.shape[2], tensor.shape[3])
            zeros = np.zeros(pad_shape, dtype=tensor.dtype)
            return np.concatenate((tensor, zeros), axis=0)
        return tensor

# =====================================================================
# 2. A REDE NEURAL (LATE FUSION)
# =====================================================================
class CNN_DualBranch(nn.Module):
    def __init__(self, canais_10m=96, canais_20m=144, num_classes=2):
        super(CNN_DualBranch, self).__init__()
        
        # Ramo 1: Resolução Espacial (10m)
        self.ramo_10m = nn.Sequential(
            nn.Conv2d(canais_10m, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Ramo 2: Espectro (20m)
        self.ramo_20m = nn.Sequential(
            nn.Conv2d(canais_20m, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Classificador Final
        self.classificador = nn.Sequential(
            nn.Linear(128 + 128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_10m, x_20m):
        f_10m = self.ramo_10m(x_10m)
        f_20m = self.ramo_20m(x_20m)
        features_combinadas = torch.cat((f_10m, f_20m), dim=1)
        return self.classificador(features_combinadas)

# =====================================================================
# 3. O MOTOR DE TREINO E VALIDAÇÃO
# =====================================================================
def treinar_modelo():
    # ---------------------------------------------------------
    # CONFIGURAÇÕES
    # ---------------------------------------------------------
    FICHEIRO_CSV = 'dataset/dataset_index.csv'
    PASTA_TENSORES = 'dataset/Tensores_Treino/'
    BATCH_SIZE = 32
    NUM_EPOCAS = 50
    LEARNING_RATE = 0.001
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> A iniciar processamento no dispositivo: {device.type.upper()}")

    # ---------------------------------------------------------
    # PREPARAÇÃO DOS DADOS
    # ---------------------------------------------------------
    df_completo = pd.read_csv(FICHEIRO_CSV)
    print("-> A verificar integridade física dos tensores...")
    
    linhas_validas = []
    
    for _, linha in df_completo.iterrows():
        caminho_10 = os.path.join(PASTA_TENSORES, f"{linha['nome_base']}_10m.npy")
        caminho_20 = os.path.join(PASTA_TENSORES, f"{linha['nome_base']}_20m.npy")
        
        # Só aceita a linha se os dois ficheiros estiverem na pasta
        if os.path.exists(caminho_10) and os.path.exists(caminho_20):
            linhas_validas.append(True)
        else:
            linhas_validas.append(False)
            
    # Filtra o dataframe apenas com o que realmente existe no disco
    df_completo = df_completo[linhas_validas].reset_index(drop=True)
    print(f"-> Fazendas válidas encontradas no disco: {len(df_completo)}")
    # Separa 80% para Treino e 20% para Validação (teste cego)
    df_treino, df_val = train_test_split(df_completo, test_size=0.20, random_state=42, stratify=df_completo['label_ia'])
    
    dataset_treino = DatasetDualBranch(df_treino, PASTA_TENSORES)
    dataset_val = DatasetDualBranch(df_val, PASTA_TENSORES)
    
    train_loader = DataLoader(dataset_treino, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Calcula os pesos das classes para equilibrar Citrus vs Resto
    labels_treino = df_treino['label_ia'].values
    pesos = compute_class_weight('balanced', classes=np.unique(labels_treino), y=labels_treino)
    pesos_tensor = torch.tensor(pesos, dtype=torch.float32).to(device)
    
    print(f"-> Fazendas de Treino: {len(df_treino)} | Validação: {len(df_val)}")
    print(f"-> Pesos aplicados às classes: {pesos}")

    # ---------------------------------------------------------
    # INICIALIZAÇÃO DO MODELO
    # ---------------------------------------------------------
    modelo = CNN_DualBranch(num_classes=2).to(device)
    criterio = nn.CrossEntropyLoss(weight=pesos_tensor) # O peso força a IA a olhar para a minoria
    otimizador = optim.Adam(modelo.parameters(), lr=LEARNING_RATE)

    # ---------------------------------------------------------
    # LOOP DE ÉPOCAS
    # ---------------------------------------------------------
    melhor_f1 = 0.0

    for epoca in range(NUM_EPOCAS):
        # -- TREINO --
        modelo.train()
        loss_treino_acumulado = 0.0
        
        loop_treino = tqdm(train_loader, desc=f"Época {epoca+1}/{NUM_EPOCAS} [Treino]", leave=False)
        for b_10m, b_20m, labels in loop_treino:
            b_10m, b_20m, labels = b_10m.to(device), b_20m.to(device), labels.to(device)
            
            otimizador.zero_grad()
            saida = modelo(b_10m, b_20m)
            loss = criterio(saida, labels)
            
            loss.backward()
            otimizador.step()
            
            loss_treino_acumulado += loss.item()
            loop_treino.set_postfix(loss=loss.item())
            
        loss_treino_medio = loss_treino_acumulado / len(train_loader)
        
        # -- VALIDAÇÃO --
        modelo.eval()
        loss_val_acumulado = 0.0
        
        labels_reais = []
        predicoes = []
        probabilidades = []
        
        with torch.no_grad():
            loop_val = tqdm(val_loader, desc=f"Época {epoca+1}/{NUM_EPOCAS} [Valid]", leave=False)
            for b_10m, b_20m, labels in loop_val:
                b_10m, b_20m, labels = b_10m.to(device), b_20m.to(device), labels.to(device)
                
                saida = modelo(b_10m, b_20m)
                loss = criterio(saida, labels)
                loss_val_acumulado += loss.item()
                
                probs = torch.softmax(saida, dim=1)
                _, classe_predita = torch.max(probs, 1)
                
                labels_reais.extend(labels.cpu().numpy())
                predicoes.extend(classe_predita.cpu().numpy())
                probabilidades.extend(probs[:, 1].cpu().numpy()) # Probabilidade de ser classe 1 (Citrus)

        loss_val_medio = loss_val_acumulado / len(val_loader)
        
        # -- CÁLCULO DAS MÉTRICAS --
        acc = accuracy_score(labels_reais, predicoes)
        prec = precision_score(labels_reais, predicoes, zero_division=0)
        rec = recall_score(labels_reais, predicoes, zero_division=0)
        f1 = f1_score(labels_reais, predicoes, zero_division=0)
        
        # O try/except garante que a AUC não quebra se um batch de teste vier só com 1 classe
        try:
            auc = roc_auc_score(labels_reais, probabilidades)
        except ValueError:
            auc = 0.0
        
        # Imprime o resumo da época
        print(f"\n[Época {epoca+1}/{NUM_EPOCAS}] Loss Tr: {loss_treino_medio:.4f} | Loss Val: {loss_val_medio:.4f}")
        print(f"Métricas Val -> AUC: {auc:.4f} | F1: {f1:.4f} | Recall: {rec:.4f} | Precision: {prec:.4f} | Acc: {acc:.4f}")
        
        # Salva o melhor modelo baseado no F1-Score
        if f1 > melhor_f1:
            melhor_f1 = f1
            torch.save(modelo.state_dict(), 'melhor_modelo_dual_branch.pth')
            print("  ⭐ Novo melhor modelo guardado!")

    print("\nTreinamento concluído. O melhor modelo foi salvo em 'melhor_modelo_dual_branch.pth'.")

if __name__ == '__main__':
    treinar_modelo()