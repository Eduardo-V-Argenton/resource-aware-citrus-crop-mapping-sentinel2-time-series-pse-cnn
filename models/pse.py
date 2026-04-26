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
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

torch.set_float32_matmul_precision('high')

# =====================================================================
# INITIAL CONFIGURATIONS AND DIRECTORY CREATION FOR THE PAPER
# =====================================================================
INDEX_FILE = '/mnt/SSD_SATA/dataset/dataset_index.csv'
TENSORS_FOLDER = "dataset/Tensors/"  
TARGET_COLUMN = "label_ia"
INPUT_CHANNELS = 10
S_SAMPLES = 128  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch configured to use: {device}")

# Creating directory structure to save paper artifacts
BASE_RESULTS_DIR = "paper_results"
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_RESULTS_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(BASE_RESULTS_DIR, "loss_history"), exist_ok=True)
os.makedirs(os.path.join(BASE_RESULTS_DIR, "classification_reports"), exist_ok=True)
os.makedirs(os.path.join(BASE_RESULTS_DIR, "raw_predictions"), exist_ok=True)
os.makedirs(os.path.join(BASE_RESULTS_DIR, "batch_tracking"), exist_ok=True)

# -------------------------------------------------------------------------
# 1. PSE DATALOADER
# -------------------------------------------------------------------------
class PSEDataset(Dataset):
    def __init__(self, base_names_list, labels, folder, samples=64, is_train=False):
        self.base_names_list = base_names_list
        self.labels = labels
        self.folder = folder
        self.samples = samples
        self.is_train = is_train

    def __len__(self):
        return len(self.base_names_list)

    def __getitem__(self, idx):
        base_name = self.base_names_list[idx]
        label = self.labels[idx]

        # Load tensor (Time, Channels, Pixels)
        tensor = np.load(os.path.join(self.folder, f"{base_name}_pse.npy"))
        total_pixels = tensor.shape[2]

        replace = total_pixels < self.samples
        indices = np.random.choice(total_pixels, self.samples, replace=replace)
        sampled_tensor = tensor[:, :, indices]
        
        if self.is_train and np.random.rand() > 0.5:
            t_idx = np.random.randint(0, sampled_tensor.shape[0])
            sampled_tensor[t_idx, :, :] = 0.0
            
        final_tensor = torch.from_numpy(sampled_tensor.astype(np.float32))
        
        return final_tensor, torch.tensor([label], dtype=torch.float32), base_name


# -------------------------------------------------------------------------
# 2. PIXEL-SET ENCODER + TEMPORAL ARCHITECTURE
# -------------------------------------------------------------------------
class PhenologyPSE(nn.Module):
    def __init__(self, in_channels=10): 
        super(PhenologyPSE, self).__init__()

        self.input_norm = nn.LayerNorm(in_channels) 
        
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
            nn.Linear(128, 32), 
            nn.GELU(), 
            nn.Dropout(0.4), 
            nn.Linear(32, 1)
        )

    def forward(self, x):
        B, T, C, S = x.size()
        
        x = x.permute(0, 1, 3, 2) # [B, T, S, C]
        x = self.input_norm(x)
        x = x.permute(0, 1, 3, 2).reshape(B * T, C, S)
                  
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


def predict(loader, tta_steps=5):
    """Prediction function with Test-Time Augmentation (TTA)"""
    model.eval()
    y_true_final = []
    accumulated_probs = None
    
    with torch.no_grad():
        for step in range(tta_steps):
            y_true_step, probs_step = [], []
            for batch_x, batch_y, _ in loader:
                batch_probs = torch.sigmoid(model(batch_x.to(device))).cpu().numpy()
                probs_step.extend(batch_probs)
                
                if step == 0:
                    y_true_step.extend(batch_y.numpy())
            
            if step == 0:
                accumulated_probs = np.array(probs_step)
                y_true_final = np.array(y_true_step)
            else:
                accumulated_probs += np.array(probs_step)
                
    final_probs = accumulated_probs / tta_steps
    return y_true_final.flatten(), final_probs.flatten()


def extract_coordinates(id_str):
    parts = str(id_str).split('_')
    if len(parts) >= 3:
        return float(parts[1]), float(parts[2])
    return np.nan, np.nan
    
# -------------------------------------------------------------------------
# 3. THE EXPERIMENT LOOP
# -------------------------------------------------------------------------
df_index = pd.read_csv(INDEX_FILE)

df_index['lon'], df_index['lat'] = zip(*df_index['id'].apply(extract_coordinates))
DEGREE_RESOLUTION = 0.02 

# Round coordinates to "fit" points into grid boxes
df_index['lat_grid'] = np.floor(df_index['lat'] / DEGREE_RESOLUTION) * DEGREE_RESOLUTION
df_index['lon_grid'] = np.floor(df_index['lon'] / DEGREE_RESOLUTION) * DEGREE_RESOLUTION

df_index['id_cluster'] = "grid_" + df_index['lat_grid'].astype(str) + "_" + df_index['lon_grid'].astype(str)

df_index = df_index.dropna(subset=['lat', 'lon']).copy()

total_original = df_index['id'].nunique()
total_clusters = df_index['id_cluster'].nunique()
reduction = (1 - (total_clusters / total_original)) * 100

print(f"Total original IDs (Centroids): {total_original}")
print(f"Total Macro-Farms (~2km Grids): {total_clusters}")
print(f"ID sample space reduction: {reduction:.1f}%")

print("Checking physical integrity of PSE tensors...")
existing_files = set([
    f.replace("_pse.npy", "")
    for f in os.listdir(TENSORS_FOLDER) if f.endswith("_pse.npy")
])

df_index = df_index[df_index["name"].isin(existing_files)].copy()

test_years = [2024,2023]
seeds = [42, 43, 44, 45]

BATCH_SIZE = 128
EPOCHS = 100
PATIENCE = 20 

general_results = {}

for test_year in test_years:
    print(f"\n{'=' * 80}\n UNSEEN TEST: YEAR {test_year} \n{'=' * 80}")

    df_test_idx = df_index[df_index["ano"] == test_year].copy()
    val_years = [test_year - 1, test_year - 2]
    
    df_rest = df_index[~df_index["ano"].isin([test_year])].copy()

    for seed in seeds:
        print(f"\n{'-' * 50}\n RUN: Seed {seed}\n{'-' * 50}")

        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        
        train_idx, val_idx = next(gss.split(df_rest, groups=df_rest["id_cluster"]))
        
        df_train_pool = df_rest.iloc[train_idx].copy()
        df_val_pool = df_rest.iloc[val_idx].copy()
        
        df_train = df_train_pool[~df_train_pool["ano"].isin(val_years)].copy()
        df_val = df_val_pool[df_val_pool["ano"].isin(val_years)].copy()
        df_train = df_train.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        df_val = df_val.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        train_dataset = PSEDataset(
            df_train["name"].values, df_train[TARGET_COLUMN].values,
            TENSORS_FOLDER, samples=S_SAMPLES, is_train=True
        )
        val_dataset = PSEDataset(
            df_val["name"].values, df_val[TARGET_COLUMN].values,
            TENSORS_FOLDER, samples=S_SAMPLES
        )
        test_dataset = PSEDataset(
            df_test_idx["name"].values, df_test_idx[TARGET_COLUMN].values,
            TENSORS_FOLDER, samples=S_SAMPLES
        )

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

        model = PhenologyPSE(in_channels=INPUT_CHANNELS).to(device)

        num_negatives = len(df_train[df_train[TARGET_COLUMN] == 0])
        num_positives = len(df_train[df_train[TARGET_COLUMN] == 1])
        citrus_weight = torch.tensor([num_negatives / num_positives], dtype=torch.float32).to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=citrus_weight).to(device)
        
        optimizer = optim.RAdam(model.parameters(), lr=1e-4, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=15, 
            T_mult=2, 
            eta_min=1e-6
        )

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_model_wts = copy.deepcopy(model.state_dict())
        
        # Dictionary to track model evolution for generating charts
        training_history = {"epoch": [], "train_loss": [], "val_loss": []}
        
        batch_tracking = []
        
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0

            train_loop = tqdm(train_loader, desc=f"Epoch {epoch + 1:03d}/{EPOCHS} [Train]", leave=False)
            for step, (batch_x, batch_y, batch_names) in enumerate(train_loop):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                batch_tracking.append({
                    "epoch": epoch + 1,
                    "batch_id": step,
                    "phase": "train",
                    "samples": "|".join(batch_names)
                })
                
                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                
                running_loss += loss.item() * batch_x.size(0)
                train_loop.set_postfix(loss=loss.item())

            epoch_loss = running_loss / len(train_loader.dataset)

            model.eval()
            val_loss = 0.0
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch + 1:03d}/{EPOCHS} [Val]", leave=False)
            
            with torch.no_grad():
                for step, (batch_x, batch_y, batch_names) in enumerate(val_loop):
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    batch_tracking.append({
                        "epoch": epoch + 1,
                        "phase": "val",
                        "batch_id": step,
                        "samples": "|".join(batch_names)
                    })
                    loss = criterion(model(batch_x), batch_y)
                    val_loss += loss.item() * batch_x.size(0)

            val_epoch_loss = val_loss / len(val_loader.dataset)
            scheduler.step()
            
            # Saving data to history
            training_history["epoch"].append(epoch + 1)
            training_history["train_loss"].append(epoch_loss)
            training_history["val_loss"].append(val_epoch_loss)

            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
                status = " (Improved!)"
            else:
                epochs_no_improve += 1
                status = ""

            print(f"Epoch {epoch + 1:03d}/{EPOCHS} | Train: {epoch_loss:.4f} | Val: {val_epoch_loss:.4f}{status}")

            if epochs_no_improve >= PATIENCE:
                print(f"-> Early Stopping: Training stopped at epoch {epoch + 1}.")
                break

        # =====================================================================
        # EXPORTING ARTIFACTS AND RESULTS
        # =====================================================================
        print("\nLoading best weights to generate metrics and export data...")
        model.load_state_dict(best_model_wts)
        
        # 1. Save the trained model weights
        model_path = os.path.join(BASE_RESULTS_DIR, "models", f"weights_year_{test_year}_seed_{seed}.pth")
        torch.save(best_model_wts, model_path)
        
        # 2. Save learning history (Loss)
        df_history = pd.DataFrame(training_history)
        df_history.to_csv(os.path.join(BASE_RESULTS_DIR, "loss_history", f"loss_year_{test_year}_seed_{seed}.csv"), index=False)

        y_true_val, probs_val = predict(val_loader)
        y_true_test, probs_test = predict(test_loader)

        precisions, recalls, thresholds = precision_recall_curve(y_true_val, probs_val)
        # recalls = recalls[:-1]
        # precisions = precisions[:-1]
        # mask = recalls >= 0.75
        # if np.any(mask):
        #     f1_scores = 2 * (precisions[mask] * recalls[mask]) / (precisions[mask] + recalls[mask] + 1e-8)
        #     best_index = np.argmax(f1_scores)
        #     optimal_threshold = thresholds[mask][best_index]
        # else:
        #     best_index = np.argmax(recalls)
        #     optimal_threshold = thresholds[best_index]

        # print(f">> Optimized Threshold on Validation: {optimal_threshold:.4f}")
        
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[best_idx]
        
        print(f">> Threshold: {optimal_threshold:.4f}")
        preds_val = (probs_val >= optimal_threshold).astype(int)
        preds_test = (probs_test >= optimal_threshold).astype(int)

        # 3. Extract and save scikit-learn reports as CSV
        report_val_dict = classification_report(y_true_val, preds_val, zero_division=0, output_dict=True)
        report_test_dict = classification_report(y_true_test, preds_test, zero_division=0, output_dict=True)
        
        pd.DataFrame(report_val_dict).transpose().to_csv(
            os.path.join(BASE_RESULTS_DIR, "classification_reports", f"val_year_{test_year}_seed_{seed}.csv")
        )
        pd.DataFrame(report_test_dict).transpose().to_csv(
            os.path.join(BASE_RESULTS_DIR, "classification_reports", f"test_year_{test_year}_seed_{seed}.csv")
        )

        # 4. Save raw predictions (Ground Truth vs Real Probability vs Final Prediction)
        df_raw_preds = pd.DataFrame({
            "y_true": y_true_test,
            "model_probability": probs_test,
            "final_threshold_prediction": preds_test
        })
        df_raw_preds.to_csv(
            os.path.join(BASE_RESULTS_DIR, "raw_predictions", f"predictions_year_{test_year}_seed_{seed}.csv"), 
            index=False
        )
        df_batches = pd.DataFrame(batch_tracking)
        df_batches.to_csv(
            os.path.join(BASE_RESULTS_DIR, "batch_tracking", f"batch_composition_year_{test_year}_seed_{seed}.csv"), 
            index=False
        )

        print(f">> TEST Result ({test_year} - {seed}):")
        print(classification_report(y_true_test, preds_test, zero_division=0))

        df_test_current = df_test_idx.copy()
        df_test_current["AI_prediction"] = preds_test

        col_key = (test_year, seed)
        general_results[col_key] = {}

        for crop, crop_data in df_test_current.groupby("crop"):
            total = len(crop_data)
            real_target = 1 if crop == "Citrus" else 0
            hits = (crop_data["AI_prediction"] == real_target).sum()
            accuracy_rate = (hits / total) * 100
            general_results[col_key][crop] = f"{hits}/{total} ({accuracy_rate:.1f}%)"

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_test, preds_test, labels=[0, 1], zero_division=0
        )

        general_results[col_key]["Recall 0"] = round(recall[0], 2)
        general_results[col_key]["Recall 1"] = round(recall[1], 2)
        general_results[col_key]["Precision 0"] = round(precision[0], 2)
        general_results[col_key]["Precision 1"] = round(precision[1], 2)

# =====================================================================
# FINAL COMPILED TABLE GENERATION
# =====================================================================
print("\n" + "=" * 80)
print(" FINAL RESULTS TABLE (FINE ANALYSIS AND GENERAL METRICS)")
print("=" * 80)

df_table = pd.DataFrame(general_results)
df_table = df_table[sorted(df_table.columns)]

ordered_crops = sorted([c for c in df_index["crop"].dropna().unique()])
row_order = ordered_crops + ["Recall 0", "Recall 1", "Precision 0", "Precision 1"]

df_table = df_table.reindex(row_order)
df_table.columns.names = ["Year", "Seed"]

print(df_table.to_string())
final_table_path = os.path.join(BASE_RESULTS_DIR, "consolidated_paper_table.csv")
df_table.to_csv(final_table_path)
print(f"\nAll artifacts were successfully saved in the folder: {BASE_RESULTS_DIR}/")
