"""
    The six explanation techniques behind one contract (see xai.contract).
"""
from __future__ import annotations

# from explainer import AttributionResult, Explainer
from gradcam import GradCamCaptum
from xai.integrated_gradients import IntegratedGradientsCaptum
from xai.lime_explainer import LimeExplainer
from xai.occlusion import OcclusionCaptum
from xai.saliency import SaliencyCaptum
from xai.shap_explainer import ShapExplainer

TECHNIQUE_NAMES = ("gradcam", "integrated_gradients", "occlusion", "saliency", "lime", "shap")


def build_techniques(model, cfg, background=None, names=TECHNIQUE_NAMES) -> dict:
    """
        name -> Explainer, in a fixed order. 
        SHAP needs the background tensor. 
        Nothing is skipped silently: an unknown name or a missing background is an error.
    """
    unknown = [n for n in names if n not in TECHNIQUE_NAMES]
    if unknown:
        raise ValueError(f"Technique unknown; techniques available: {TECHNIQUE_NAMES}")
    if "shap" in names and background is None:
        raise ValueError("SHAP needs background tensor (xai.shap_explainer.make_background)")
    
    builders = { "gradcam": lambda: GradCamCaptum(model, cfg),
                 "integrated_gradients": lambda: IntegratedGradientsCaptum(model, cfg),
                 "occlusion": lambda: OcclusionCaptum(model, cfg),
                 "saliency": lambda: SaliencyCaptum(model, cfg),
                 "lime": lambda: LimeExplainer(model, cfg),
                 "shap": lambda: ShapExplainer(model, cfg, background),
    }

    return {n: builders[n]() for n in TECHNIQUE_NAMES if n in names}