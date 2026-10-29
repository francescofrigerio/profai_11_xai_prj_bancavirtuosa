| Check | Status | Detail |
|---|---|---|
| split sizes | PASS | train 54000, validation 6000, test 10000 |
| split is disjoint | PASS | no image in both sets |
| model mode | INFO | load (compatible checkpoint found) |
| checkpoint round trip | PASS | reloaded validation accuracy 0.9700 vs training run 0.9700 |
| DenseNet validation accuracy >= MIN_VAL_ACCURACY | PASS | 0.9700 vs 0.95 |
| LR converged | PASS | 248 iterations (LR_MAX_ITER=1000) |
| LR validation accuracy in the expected range | PASS | 0.9222 (design expects about 0.92) |
| LR logit is exactly w.x + b | PASS | max error 0.0e+00 |
| LR coefficients sum to ~0 over the classes | PASS | max 1.0e-13 (l2: relative = absolute coefficients) |
| LR classes aligned with the digits | PASS | classes_ == 0..9 |
| DenseNet beats the LR on the test set | PASS | test accuracy 0.9708 vs 0.9246 |
| gap between the models as the design expects | PASS | 4.62 points (design: 4-6 points on validation) |
| the four quadrants sum to the test size | PASS | {'both_correct': 9071, 'only_densenet': 637, 'only_lr': 175, 'both_wrong': 117} |
