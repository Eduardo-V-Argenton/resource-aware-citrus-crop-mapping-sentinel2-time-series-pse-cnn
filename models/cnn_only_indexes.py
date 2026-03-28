import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report ,precision_recall_fscore_support
import copy
from tqdm import tqdm
import torchvision.transforms as T
from torch.utils.data import WeightedRandomSampler

FICHEIRO_INDEX = 'dataset/dataset_index.csv'
PASTA_TENSORES = 'dataset/Tensores_Indexes_FFT/'
COLUNA_ALVO = 'label_ia'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch configurado para usar: {device}")

# -------------------------------------------------------------------------
# 1. DATALOADER DIRETO E RÁPIDO (Ajustado para 12 canais)
# -------------------------------------------------------------------------
class FastFFTDataset(Dataset):
    def __init__(self, lista_nomes_base, labels, pasta, is_train=False):
        self.lista_nomes_base = lista_nomes_base
        self.labels = labels
        self.pasta = pasta
        self.is_train = is_train 

        self.augmentations = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ])
        
    def __len__(self):
        return len(self.lista_nomes_base)

    def __getitem__(self, idx):
        nome_base = self.lista_nomes_base[idx]
        label = self.labels[idx]
        
        tensor_fft = np.load(os.path.join(self.pasta, f"{nome_base}_fft.npy"))
        tensor_final = torch.from_numpy(tensor_fft)
        if self.is_train:
            tensor_final = self.augmentations(tensor_final)
            
        return tensor_final, torch.tensor([label], dtype=torch.float32)
            

# -------------------------------------------------------------------------
# 2. ARQUITETURA CNN 2D OTIMIZADA (Ajustada para in_channels=12)
# -------------------------------------------------------------------------
class PhenologyCNN2D(nn.Module):
    def __init__(self, in_channels=12):
        super(PhenologyCNN2D, self).__init__()
        
        self.features = nn.Sequential(
            nn.InstanceNorm2d(in_channels, affine=True),
            
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.InstanceNorm2d(16, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def prever(loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            preds = (torch.sigmoid(model(batch_x.to(device))) > 0.5).int().cpu().numpy()
            y_true.extend(batch_y.numpy()), y_pred.extend(preds)
    return np.array(y_true).flatten(), np.array(y_pred).flatten()

# -------------------------------------------------------------------------
# 3. O LOOP DO EXPERIMENTO
# -------------------------------------------------------------------------
# =====================================================================
# CONFIGURAÇÕES INICIAIS
# =====================================================================
df_index = pd.read_csv(FICHEIRO_INDEX)
df_index = df_index[df_index['ano'] >= 2019]

anos_teste = [2024, 2023, 2022]
seeds = [42, 43, 44]

BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 10

# Dicionário mestre para guardar os resultados de TODAS as rodadas
resultados_gerais = {}

for ano_teste in anos_teste:
    print(f"\n{'='*80}\n TESTE INÉDITO: ANO {ano_teste} \n{'='*80}")
    
    # IMPORTANTE: Usamos .copy() para evitar o SettingWithCopyWarning do Pandas
    df_test_idx = df_index[df_index['ano'] == ano_teste].copy()
    df_resto_idx = df_index[df_index['ano'] != ano_teste].copy()
    
    for seed in seeds:
        print(f"\n{'-'*50}\n RODADA: Seed {seed}\n{'-'*50}")
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available(): 
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
        df_train, df_val = train_test_split(df_resto_idx, test_size=0.1765, random_state=seed, stratify=df_resto_idx[COLUNA_ALVO])
        
        train_dataset = FastFFTDataset(df_train['nome_base'].values, df_train[COLUNA_ALVO].values, PASTA_TENSORES, is_train=True)
        val_dataset = FastFFTDataset(df_val['nome_base'].values, df_val[COLUNA_ALVO].values, PASTA_TENSORES, is_train=False)
        test_dataset = FastFFTDataset(df_test_idx['nome_base'].values, df_test_idx[COLUNA_ALVO].values, PASTA_TENSORES, is_train=False)
        
        # 1. Conta a Laranja (Classe 1)
        qtd_laranja = len(df_train[df_train['label_ia'] == 1])
        
        # 2. Descobre quantas culturas diferentes formam a "Salada" (Classe 0)
        culturas_classe_0 = df_train[df_train['label_ia'] == 0]['classe_mapbiomas'].unique()
        num_culturas_0 = len(culturas_classe_0)
        
        # 3. Cria o dicionário de pesos matemáticos
        pesos_dit = {}
        pesos_dit[47] = 0.5 / qtd_laranja
        
        for cultura_id in culturas_classe_0:
            qtd_desta_cultura = len(df_train[df_train['classe_mapbiomas'] == cultura_id])
            pesos_dit[cultura_id] = (0.5 / num_culturas_0) / qtd_desta_cultura
        
        # 4. Mapeia o peso para cada polígono individual do dataframe de treino
        pesos_das_amostras = df_train['classe_mapbiomas'].map(pesos_dit).values
        pesos_tensor = torch.DoubleTensor(pesos_das_amostras)
        
        # 5. Instancia a Roleta Viciada (Sampler)
        sampler_treino = WeightedRandomSampler(
            weights=pesos_tensor,
            num_samples=len(pesos_tensor),
            replacement=True
        )
        
        train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler_treino)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Instancia o modelo
        model = PhenologyCNN2D(in_channels=15).to(device) 
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_wts = copy.deepcopy(model.state_dict())
        
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            
            loop_treino = tqdm(train_loader, desc=f"Época {epoch+1:03d}/{EPOCHS} [Treino]", leave=False)
            for batch_x, batch_y in loop_treino:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += loss.item() * batch_x.size(0)
                loop_treino.set_postfix(loss=loss.item())
            
            epoch_loss = running_loss / len(train_loader.dataset)
            
            model.eval()
            val_loss = 0.0
            loop_val = tqdm(val_loader, desc=f"Época {epoch+1:03d}/{EPOCHS} [Valida]", leave=False)
            with torch.no_grad():
                for batch_x, batch_y in loop_val:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    loss = criterion(model(batch_x), batch_y)
                    val_loss += loss.item() * batch_x.size(0)
            
            val_epoch_loss = val_loss / len(val_loader.dataset)
            scheduler.step(val_epoch_loss)
            
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
                status = " (Melhorou!)"
            else:
                epochs_no_improve += 1
                status = ""
            
            print(f"Época {epoch+1:03d}/{EPOCHS} | Train: {epoch_loss:.4f} | Val: {val_epoch_loss:.4f}{status}")
                
            if epochs_no_improve >= PATIENCE:
                print(f"-> Early Stopping: O treinamento parou na época {epoch+1}.")
                break
                
        # =====================================================================
        # EXTRAÇÃO DE RESULTADOS E AUDITORIA (NO CONJUNTO DE TESTE)
        # =====================================================================
        print("\nCarregando os melhores pesos para gerar as métricas...")
        model.load_state_dict(best_model_wts)
        

        y_true_val, preds_val = prever(val_loader)
        y_true_test, preds_test = prever(test_loader)
        
        print(f">> Resultado no VAL ({ano_teste} - {seed}):")
        print(classification_report(y_true_val, preds_val, zero_division=0))
        print(f">> Resultado no TESTE ({ano_teste} - {seed}):")
        print(classification_report(y_true_test, preds_test, zero_division=0))
        
        # Cola as previsões no DataFrame do Ano de Teste Atual
        df_test_current = df_test_idx.copy()
        df_test_current['previsao_IA'] = preds_test
        
        # Prepara a chave para salvar no dicionário (Ano, Seed)
        col_key = (ano_teste, seed)
        resultados_gerais[col_key] = {}
        
        # 1. Auditoria de Cultura por Cultura
        for cultura, dados_cultura in df_test_current.groupby('cultura_real'):
            total = len(dados_cultura)
            alvo_real = 1 if cultura == 'Citrus' else 0
            
            acertos = (dados_cultura['previsao_IA'] == alvo_real).sum()
            taxa_acerto = (acertos / total) * 100
            
            # Formato da string exatamente como na imagem: "16/22 (72.7%)"
            resultados_gerais[col_key][cultura] = f"{acertos}/{total} ({taxa_acerto:.1f}%)"
            
        # 2. Extrai Precision e Recall exatos do Scikit-Learn
        precision, recall, f1, _ = precision_recall_fscore_support(y_true_test, preds_test, labels=[0, 1], zero_division=0)
        
        resultados_gerais[col_key]['Recall 0'] = round(recall[0], 2)
        resultados_gerais[col_key]['Recall 1'] = round(recall[1], 2)
        resultados_gerais[col_key]['Precision 0'] = round(precision[0], 2)
        resultados_gerais[col_key]['Precision 1'] = round(precision[1], 2)
        
        print(f">> Resultado no TESTE (Ano Inédito {ano_teste} | Seed {seed}): Auditado e Salvo!")

# =====================================================================
# GERAÇÃO DA TABELA FINAL COMPILADA
# =====================================================================
print("\n" + "="*80)
print(" TABELA FINAL DE RESULTADOS (ANÁLISE FINA E MÉTRICAS GERAIS)")
print("="*80)

# Transforma o dicionário em DataFrame
df_tabela = pd.DataFrame(resultados_gerais)

# Ordena as colunas cronologicamente e por seed (Ex: 2022 -> 2023 -> 2024)
df_tabela = df_tabela[sorted(df_tabela.columns)]

# Puxa o nome de todas as culturas que apareceram para garantir a ordem correta
culturas_ordenadas = sorted([c for c in df_index['cultura_real'].dropna().unique()])

# Junta com as métricas no fundo (exatamente como na imagem)
ordem_linhas = culturas_ordenadas + ['Recall 0', 'Recall 1', 'Precision 0', 'Precision 1']

# Reordena o índice do DataFrame
df_tabela = df_tabela.reindex(ordem_linhas)

# Nomeia os eixos do cabeçalho
df_tabela.columns.names = ['Ano', 'Seed']

print(df_tabela.to_string())

df_tabela.to_csv("auditoria_completa_mestrado.csv")