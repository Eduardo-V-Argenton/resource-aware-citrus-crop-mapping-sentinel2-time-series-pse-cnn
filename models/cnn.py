import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import copy
import warnings

# Ignora avisos do Scipy sobre zoom
warnings.filterwarnings(action='ignore', category=UserWarning, module='scipy.ndimage')

# Configurações de Caminhos e Dados
PASTA_TENSORES = 'dataset/Tensores_FFT_Prontos/'
FICHEIRO_INDEX = 'dataset/dataset_index.csv'
COLUNA_ALVO = 'label_ia'

# Configuração de Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando processamento via: {device}")

# -------------------------------------------------------------------------
# 1. DATALOADER COM COMPRESSÃO HARMÔNICA (FFT)
# -------------------------------------------------------------------------
class FFTCNN2DDataset(Dataset):
    def __init__(self, lista_nomes_base, labels, pasta_tensores_prontos):
        self.lista_nomes_base = lista_nomes_base
        self.labels = labels
        self.pasta_tensores = pasta_tensores_prontos # Aponta para a NOVA pasta

    def __len__(self):
        return len(self.lista_nomes_base)

    def __getitem__(self, idx):
        nome_base = self.lista_nomes_base[idx]
        label = self.labels[idx]
        
        try:
            # Carrega direto o arquivo de 30 canais já calculado e com borda
            caminho_arquivo = os.path.join(self.pasta_tensores, f"{nome_base}_fft.npy")
            tensor_fft = np.load(caminho_arquivo)
            
            input_tensor = torch.from_numpy(tensor_fft)
            label_t = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
            
            return input_tensor, label_t
            
        except Exception as e:
            # Caso algum arquivo falte
            return torch.zeros((30, 168, 168)), torch.tensor([0.0])

# -------------------------------------------------------------------------
# 2. ARQUITETURA DA CNN 2D (Super Leve)
# -------------------------------------------------------------------------
class HarmonicCNN2D(nn.Module):
    def __init__(self, in_channels=30): # 3 coeficientes * 10 bandas brutas
        super(HarmonicCNN2D, self).__init__()
        
        # Bloco 1
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2) # Reduz H e W
        
        # Bloco 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bloco 3 (Extração espacial profunda das ruas do pomar)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Esmaga o tensor espacial para um vetor único de 256 características
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Camadas Densas Finais
        self.fc1 = nn.Linear(256, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = torch.relu(self.bn3(self.conv3(x)))
        
        x = self.global_pool(x).squeeze(-1).squeeze(-1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

# -------------------------------------------------------------------------
# 3. LOOP DE EXPERIMENTOS
# -------------------------------------------------------------------------
print("Preparando índice de dados...")
df_index = pd.read_csv(FICHEIRO_INDEX)
df_index = df_index[df_index['ano'] >= 2019]

anos_teste = [2024, 2023, 2022]
seeds = [42, 43, 44]

# Como o modelo é leve, podemos usar um Batch Size maior, acelerando o treino
BATCH_SIZE = 32 
EPOCHS = 100
PATIENCE = 15

for ano_teste in anos_teste:
    print("\n" + "="*80)
    print(f" EXPERIMENTO: CNN HARMÔNICA 2D (DADOS PUROS + FFT) - ANO TESTE {ano_teste} ")
    print("="*80)
    
    df_test_idx = df_index[df_index['ano'] == ano_teste]
    df_resto_idx = df_index[df_index['ano'] != ano_teste]
    
    for seed in seeds:
        print(f"\n" + "-"*50)
        print(f" RUN: Seed {seed}")
        print("-"*50)
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        df_train_idx, df_val_idx = train_test_split(
            df_resto_idx, test_size=0.1765, random_state=seed, stratify=df_resto_idx[COLUNA_ALVO]
        )
        
        train_ds = FFTCNN2DDataset(df_train_idx['nome_base'].values, df_train_idx[COLUNA_ALVO].values, PASTA_TENSORES)
        val_ds = FFTCNN2DDataset(df_val_idx['nome_base'].values, df_val_idx[COLUNA_ALVO].values, PASTA_TENSORES)
        test_ds = FFTCNN2DDataset(df_test_idx['nome_base'].values, df_test_idx[COLUNA_ALVO].values, PASTA_TENSORES,)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        
        model = HarmonicCNN2D(in_channels=30).to(device)
        
        # Pesos para balanceamento
        pos_count = sum(df_train_idx[COLUNA_ALVO] == 1)
        neg_count = sum(df_train_idx[COLUNA_ALVO] == 0)
        pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32).to(device)
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_wts = copy.deepcopy(model.state_dict())
        
        print("Iniciando Treinamento da CNN 2D...")
        for epoch in range(EPOCHS):
            # -------------------------------------------------------------
            # FASE DE TREINO
            # -------------------------------------------------------------
            model.train()
            running_loss = 0.0
            
            # Envolve o train_loader no tqdm
            loop_treino = tqdm(train_loader, desc=f"Época {epoch+1}/{EPOCHS} [Treino]", leave=False)
            
            for batch_x, batch_y in loop_treino:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * batch_x.size(0)
                
                # Atualiza a barra com o valor do erro em tempo real
                loop_treino.set_postfix(loss=loss.item())
            
            epoch_loss = running_loss / len(train_loader.dataset)
            
            # -------------------------------------------------------------
            # FASE DE VALIDAÇÃO
            # -------------------------------------------------------------
            model.eval()
            val_running_loss = 0.0
            
            # Envolve o val_loader no tqdm
            loop_val = tqdm(val_loader, desc=f"Época {epoch+1}/{EPOCHS} [Validação]", leave=False)
            
            with torch.no_grad():
                for batch_x, batch_y in loop_val:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_running_loss += loss.item() * batch_x.size(0)
                    
                    loop_val.set_postfix(val_loss=loss.item())
            
            val_epoch_loss = val_running_loss / len(val_loader.dataset)
            
            # -------------------------------------------------------------
            # CHECAGEM DO EARLY STOPPING E PRINT FINAL DA ÉPOCA
            # -------------------------------------------------------------
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
                status_melhoria = " (Melhorou!)"
            else:
                epochs_no_improve += 1
                status_melhoria = ""
            
            # Imprime o resumo da época quando a barra terminar
            print(f"Época {epoch+1}/{EPOCHS} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f}{status_melhoria}")
                
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly Stopping acionado na época {epoch+1}. O modelo parou de aprender.")
                break
                
        model.load_state_dict(best_model_wts)
        
        def prever(loader):
            model.eval()
            y_true = []
            y_pred = []
            with torch.no_grad():
                for batch_x, batch_y in loader:
                    batch_x = batch_x.to(device)
                    logits = model(batch_x)
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).int().cpu().numpy()
                    y_true.extend(batch_y.numpy())
                    y_pred.extend(preds)
            return np.array(y_true).flatten(), np.array(y_pred).flatten()

        _, preds_val = prever(val_loader)
        y_true_test, preds_test = prever(test_loader)
        
        print(f"\n>> Desempenho na VALIDAÇÃO (Anos Misturados - Seed {seed}):")
        print(classification_report(df_val_idx[COLUNA_ALVO].values, preds_val, zero_division=0))
        
        print(f">> Desempenho no TESTE (Ano Inédito {ano_teste} - Seed {seed}):")
        print(classification_report(y_true_test, preds_test, zero_division=0))