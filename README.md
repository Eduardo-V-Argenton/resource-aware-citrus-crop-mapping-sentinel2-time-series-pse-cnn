# Resource-Aware Citrus Crop Mapping from Sentinel-2 Time Series Using a Pixel-Set Encoder Convolutional Neural Network for Sustainable Agricultural Monitoring

![Graphical Abstract](figures/ga.png)

This repository contains the codebase, generated charts, and evaluation metrics for the study on citrus crop classification using Sentinel-2 satellite image time series and MapBiomas data.

## 📖 Abstract

**Context:** Accurate citrus crop mapping is essential for agricultural monitoring, production planning, and supply-chain management, particularly in Brazil, one of the world's leading orange producers and the leading orange-juice exporter. Satellite image time series from Sentinel-2 provide rich spectral and temporal information for crop identification. However, citrus mapping remains challenging due to fragmented agricultural landscapes, cloud contamination, class imbalance, and spectral overlap with other vegetation classes. 

**Problem:** Conventional machine learning models often depend on handcrafted vegetation indices, while attention-based deep learning models may require larger datasets and can become unstable under geographically constrained conditions. Therefore, there is a need for a lightweight and robust architecture capable of extracting citrus phenological signatures directly from multispectral time-series data. 

**Methods:** This study evaluates a Spatio-Temporal Pixel-Set Encoder Convolutional Neural Network (PSE-CNN) for citrus crop classification in the immediate geographic regions of São João da Boa Vista and Mogi Guaçu, São Paulo, Brazil. MapBiomas Collection 10.1 data from 2019 to 2024 were used to derive reference polygons, and Sentinel-2 imagery was processed into cloud-masked, 15-day temporal composites using ten spectral bands. The proposed PSE-CNN was benchmarked against PSE-TAE, PSE-Transformer, Random Forest, and XGBoost using spatially grouped data partitioning and temporal test years. 

**Results:** The proposed PSE-CNN achieved the highest Unified F1-Score of **0.703** and the lowest coefficient of variation of **2.28%**, indicating stronger inter-annual stability across test years and random seeds among the evaluated models. It also outperformed classical models that relied on handcrafted vegetation indices and demonstrated greater overall stability than attention-based deep learning alternatives.

**Conclusions:** The results indicate that combining pixel-set encoding with temporal convolution provides a resource-aware and stable framework for retrospective citrus crop mapping from Sentinel-2 satellite image time series. These findings suggest that PSE-CNN can support scalable agricultural monitoring, contributing to sustainable crop inventory systems in regions where labeled data and computational infrastructure are limited.

---

## 📊 Key Results

The model's stable temporal predictive behavior are detailed in the comprehensive metrics dashboard:

![Comprehensive Metrics Dashboard](figures/charts/exported/comprehensive_metrics_dashboard.png)
