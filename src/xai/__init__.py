"""
    The six explanation techniques behind one contract (see xai.contract).
"""
from __future__ import annotations

from xai.explainer import AttributionResult, Explainer
from xai.gradcam import GradCam
from xai.integrated_gradients import IntegratedGradients
from xai.lime_explainer import Lime
from xai.occlusion import Occlusion
from xai.saliency import Saliency
from xai.shap_explainer import Shap

TECHNIQUE_NAMES = ("gradcam", "integrated_gradients", "occlusion", "saliency", "lime", "shap")


def build_techniques(model, cfg, background=None, names=TECHNIQUE_NAMES) -> dict:
    """
        name -> Explainer, in a fixed order. 
        SHAP needs the background tensor. 
        Nothing is skipped silently: an unknown name or a missing background is an error.
    """
    unknown = [n for n in names if n not in TECHNIQUE_NAMES]
    if unknown:
        raise ValueError(f"Tecnica sconosciuta {unknown}; Tecniche disponibili: {TECHNIQUE_NAMES}")
    if "shap" in names and background is None:
        raise ValueError("SHAP necessita di un tensore di background (xai.shap_explainer.make_background)")
    
    builders = { "gradcam": lambda: GradCam(model, cfg),
                 "integrated_gradients": lambda: IntegratedGradients(model, cfg),
                 "occlusion": lambda: Occlusion(model, cfg),
                 "saliency": lambda: Saliency(model, cfg),
                 "lime": lambda: Lime(model, cfg),
                 "shap": lambda: Shap(model, cfg, background),
    }

    return {n: builders[n]() for n in TECHNIQUE_NAMES if n in names}