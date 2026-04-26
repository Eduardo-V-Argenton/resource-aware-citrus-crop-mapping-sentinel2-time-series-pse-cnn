import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupShuffleSplit
import joblib

INDEX_FILE = '/mnt/SSD_SATA/dataset/dataset_index.csv'
# ML_FEATURES_FILE = 'dataset/dataset_ml_bands_indexes.csv' 
ML_FEATURES_FILE = 'dataset/dataset_ml.csv' 
TARGET_COLUMN = "label_ia"

# BASE_RESULTS_DIR = "paper_results_xgb_bands_indexes"
BASE_RESULTS_DIR = "paper_results_xgb"
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_RESULTS_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(BASE_RESULTS_DIR, "classification_reports"), exist_ok=True)
os.makedirs(os.path.join(BASE_RESULTS_DIR, "raw_predictions"), exist_ok=True)
os.makedirs(os.path.join(BASE_RESULTS_DIR, "feature_importance"), exist_ok=True)

def extract_coordinates(id_str):
    parts = str(id_str).split('_')
    if len(parts) >= 3:
        return float(parts[1]), float(parts[2])
    return np.nan, np.nan
    
# -------------------------------------------------------------------------
# 1. LOAD AND MERGE DATASETS
# -------------------------------------------------------------------------
print("Loading Metadata and Tabular Features...")
df_index = pd.read_csv(INDEX_FILE)
df_features = pd.read_csv(ML_FEATURES_FILE)

df_merged = pd.merge(df_index, df_features, left_on='name', right_on='image_name', how='inner')

feature_cols = [c for c in df_features.columns if c != 'image_name']

df_merged['lon'], df_merged['lat'] = zip(*df_merged['id'].apply(extract_coordinates))
DEGREE_RESOLUTION = 0.02 

df_merged['lat_grid'] = np.floor(df_merged['lat'] / DEGREE_RESOLUTION) * DEGREE_RESOLUTION
df_merged['lon_grid'] = np.floor(df_merged['lon'] / DEGREE_RESOLUTION) * DEGREE_RESOLUTION
df_merged['id_cluster'] = "grid_" + df_merged['lat_grid'].astype(str) + "_" + df_merged['lon_grid'].astype(str)
df_merged = df_merged.dropna(subset=['lat', 'lon']).copy()

total_original = df_merged['id'].nunique()
total_clusters = df_merged['id_cluster'].nunique()
reduction = (1 - (total_clusters / total_original)) * 100

print(f"Total original IDs (Centroids): {total_original}")
print(f"Total Macro-Farms (~2km Grids): {total_clusters}")
print(f"ID sample space reduction: {reduction:.1f}%")
print(f"Total Features available for XGBoost: {len(feature_cols)}\n")

test_years = [2024, 2023]
seeds = [42, 43, 44, 45]

general_results = {}

for test_year in test_years:
    print(f"\n{'=' * 80}\n UNSEEN TEST: YEAR {test_year} \n{'=' * 80}")

    df_test_idx = df_merged[df_merged["ano"] == test_year].copy()
    val_years = [test_year - 1, test_year - 2]
    df_rest = df_merged[~df_merged["ano"].isin([test_year])].copy()

    for seed in seeds:
        print(f"\n{'-' * 50}\n RUN: Seed {seed} (XGBoost)\n{'-' * 50}")

        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        train_idx, val_idx = next(gss.split(df_rest, groups=df_rest["id_cluster"]))
        
        df_train_pool = df_rest.iloc[train_idx].copy()
        df_val_pool = df_rest.iloc[val_idx].copy()
        
        df_train = df_train_pool[~df_train_pool["ano"].isin(val_years)].copy()
        df_val = df_val_pool[df_val_pool["ano"].isin(val_years)].copy()

        # Preparing X and y matrices
        X_train = df_train[feature_cols].values
        y_train = df_train[TARGET_COLUMN].values
        
        X_val = df_val[feature_cols].values
        y_val = df_val[TARGET_COLUMN].values
        
        X_test = df_test_idx[feature_cols].values
        y_test = df_test_idx[TARGET_COLUMN].values

        # ---------------------------------------------------------------------
        # XGBOOST CONFIGURATION & CLASS WEIGHTING
        # ---------------------------------------------------------------------
        num_negatives = (y_train == 0).sum()
        num_positives = (y_train == 1).sum()
        peso_citrus = num_negatives / num_positives

        print(f"Training XGBoost on {len(X_train)} samples (scale_pos_weight: {peso_citrus:.2f})...")
        
        model = XGBClassifier(
            n_estimators=500, 
            max_depth=8,             
            learning_rate=0.1,       
            scale_pos_weight=peso_citrus, 
            random_state=seed, 
            n_jobs=-1,
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)

        probs_val = model.predict_proba(X_val)[:, 1]
        probs_test = model.predict_proba(X_test)[:, 1]

        precisions, recalls, thresholds = precision_recall_curve(y_val, probs_val)
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

        model_path = os.path.join(BASE_RESULTS_DIR, "models", f"xgb_model_year_{test_year}_seed_{seed}.joblib")
        joblib.dump(model, model_path)

        importances = model.feature_importances_
        df_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        df_importance.to_csv(
            os.path.join(BASE_RESULTS_DIR, "feature_importance", f"importance_year_{test_year}_seed_{seed}.csv"), 
            index=False
        )

        report_val_dict = classification_report(y_val, preds_val, zero_division=0, output_dict=True)
        report_test_dict = classification_report(y_test, preds_test, zero_division=0, output_dict=True)
        
        pd.DataFrame(report_val_dict).transpose().to_csv(
            os.path.join(BASE_RESULTS_DIR, "classification_reports", f"val_year_{test_year}_seed_{seed}.csv")
        )
        pd.DataFrame(report_test_dict).transpose().to_csv(
            os.path.join(BASE_RESULTS_DIR, "classification_reports", f"test_year_{test_year}_seed_{seed}.csv")
        )

        df_raw_preds = pd.DataFrame({
            "image_name": df_test_idx["name"].values,
            "y_true": y_test,
            "model_probability": probs_test,
            "final_threshold_prediction": preds_test
        })
        df_raw_preds.to_csv(
            os.path.join(BASE_RESULTS_DIR, "raw_predictions", f"predictions_year_{test_year}_seed_{seed}.csv"), 
            index=False
        )

        print(f">> TEST Result ({test_year} - {seed}):")
        print(classification_report(y_test, preds_test, zero_division=0))

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
            y_test, preds_test, labels=[0, 1], zero_division=0
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

ordered_crops = sorted([c for c in df_merged["crop"].dropna().unique()])
row_order = ordered_crops + ["Recall 0", "Recall 1", "Precision 0", "Precision 1"]

df_table = df_table.reindex(row_order)
df_table.columns.names = ["Year", "Seed"]

print(df_table.to_string())
final_table_path = os.path.join(BASE_RESULTS_DIR, "consolidated_paper_table.csv")
df_table.to_csv(final_table_path)
print(f"\nAll artifacts were successfully saved in the folder: {BASE_RESULTS_DIR}/")