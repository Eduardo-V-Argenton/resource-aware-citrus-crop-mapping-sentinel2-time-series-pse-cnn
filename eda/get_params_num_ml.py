import joblib

total_rf_bands = 0
total_rf_idx = 0
for year in [2023,2024]:
    for seed in [42,43,44,45]:
        model = joblib.load(f"results/recall_free/paper_results_rf/models/rf_model_year_{year}_seed_{seed}.joblib")
        total_rf_bands += sum(tree.tree_.node_count for tree in model.estimators_)
        
        model = joblib.load(f"results/recall_free/paper_results_rf_bands_indexes/models/rf_model_year_{year}_seed_{seed}.joblib")
        total_rf_idx += sum(tree.tree_.node_count for tree in model.estimators_)

print(f"RF: {total_rf_bands/8} RF_idx: {total_rf_idx/8}")

total_xgb_bands = 0
total_xgb_idx = 0
for year in [2023,2024]:
    for seed in [42,43,44,45]:
        model = joblib.load(f"results/recall_free/paper_results_xgb/models/xgb_model_year_{year}_seed_{seed}.joblib")
        booster = model.get_booster()
        
        df = booster.trees_to_dataframe()
        
        total_xgb_bands += len(df)
        
        model = joblib.load(f"results/recall_free/paper_results_xgb_bands_indexes/models/xgb_model_year_{year}_seed_{seed}.joblib")
        booster = model.get_booster()
        
        df = booster.trees_to_dataframe()
        
        total_xgb_idx += len(df)

print(f"xgb: {total_xgb_bands/8} xgb_idx: {total_xgb_idx/8}")