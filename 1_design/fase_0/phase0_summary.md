# Phase 0 - table of fixed values

| Parameter | Initial | Proposed | Basis | Confirmation |
|---|---|---|---|---|
| LIME_BACKEND | lime | lime | check 3 | criterion met (automatic) |
| SHAP_BACKEND | shap | shap | check 3 | criterion met (automatic) |
| N_QUANT | 100 | 100 | check 9 (time budget) | criterion met (automatic) |
| NUM_EPOCHS | 5 | 5 | check 6: last-epoch val-loss drop 11.6% | operator confirmation required |
| MIN_VAL_ACCURACY | 0.95 | 0.95 | check 6: final-epoch validation accuracy 0.9700 | criterion met (automatic) |
| BATCH_SIZE | 64 | 64 | check 6 (60s/epoch) | criterion met (automatic) |
| LEARNING_RATE_HEAD | 0.001 | 0.001 | check 6 (60s/epoch) | criterion met (automatic) |
| FLAG_BATCHNORM_EVAL | True | True | check 6: configured 0.9700 vs opposite 0.9493 | criterion met (automatic) |
| FLAG_RELU_INPLACE | True | True | check 2: in-place model fails or gives different maps (see check 2) | FAIL: set FLAG_RELU_INPLACE=False |
| LR_C | 1.0 | 1.0 | check 5: val acc 0.9222, maps to review | operator confirmation required |
| LR_MAX_ITER | 1000 | 1000 | check 5: n_iter 248, ConvergenceWarning=False | criterion met (automatic) |
| IG_STEPS | 50 | 50 | check 4: completeness error 20.27% | operator confirmation required |
| OCCLUSION_WINDOW/STRIDE | 16/8 | 16/8 | check 11: 226 forwards, overlap averaged | criterion met (automatic) |
| SHAP_NUM_SAMPLES | 200 | 200 | check 3 timing | criterion met (automatic) |
| SHAP_NUM_BACKGROUND | 50 | 50 | check 3 timing | criterion met (automatic) |
| LIME_NUM_SAMPLES | 1000 | 1000 | check 3 timing | criterion met (automatic) |
| LIME_GRID_CELLS | 8 | 8 | check 7 (visual) | operator confirmation required |
| MASK_THRESHOLD / MASK_DILATION | 0.1/1 | 0.1/1 | check 8 (visual) | operator confirmation required |
| 28x28 resampling rule (E03) | bilinear->128->area | bilinear->128->area | check 12 (numeric + visual) | operator confirmation required |
