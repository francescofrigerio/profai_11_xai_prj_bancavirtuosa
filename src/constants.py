# Statistiche standard ImageNet
IMAGENET_MEAN_NP = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD_NP  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMAGENET_MEAN_TRANSFORM  =  [0.485, 0.456, 0.406]
IMAGENET_STD_TRANSFORM =   [0.229, 0.224, 0.225]

TITLES = {"gradcam": "Grad-CAM", "integrated_gradients": "Integrated\nGradients", "occlusion": "Occlusion",
          "saliency": "Saliency", "lime": "LIME", "shap": "SHAP"}

GROUPS_RESULT_OK_KO = { "both_ok":      "LR ok / CNN ok",
                        "lr_ko_cnn_ok": "LR errata / CNN ok",
                        "lr_ok_cnn_ko": "LR ok / CNN errata",
                        "both_ko":      "LR errata / CNN errata"
                        }
