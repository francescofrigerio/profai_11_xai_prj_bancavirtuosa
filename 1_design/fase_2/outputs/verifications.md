| Check | Status | Detail |
|---|---|---|
| model mode | INFO | train (no checkpoint at checkpoints/densenet121_mnist.pt); checkpoint sha256 b503dddb4071a5dc... |
| techniques run | PASS | 6 techniques on 10 images, every map finite, 2D and within the contract (the contract raises otherwise) |
| gradcam: time per image | PASS | median 0.02s (limit 15s) |
| integrated_gradients: time per image | PASS | median 0.51s (limit 15s) |
| occlusion: time per image | PASS | median 0.26s (limit 15s) |
| saliency: time per image | PASS | median 0.04s (limit 15s) |
| lime: time per image | PASS | median 3.20s (limit 15s) |
| shap: time per image | PASS | median 3.36s (limit 15s) |
| Integrated Gradients completeness | PASS | worst relative error 2.70% over 10 images (limit 5%) |
| lime is reproducible | PASS | same seed, same map on the first image |
| shap is reproducible | PASS | same seed, same map on the first image |
| in-place ReLU gives the same gradcam map on the trained model | PASS | relative difference 0.0e+00 |
| in-place ReLU gives the same integrated_gradients map on the trained model | PASS | relative difference 1.6e-07 |
| in-place ReLU gives the same saliency map on the trained model | PASS | relative difference 2.5e-07 |
| saliency on the digit (information) | INFO | 32% of its mass inside the digit mask, a random map would give 33% |
