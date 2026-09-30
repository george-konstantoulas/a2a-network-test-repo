from truth_predictor.zero_shot_predictor import ZeroShotPredictor
from truth_predictor.fine_tuned_predictor import FineTunedPredictor
from truth_predictor.explainer import Explainer
from mcp_package.schemas.schemas import ExplanationInput
from mcp_package.config import config
from truth_predictor.utils import load_from_user_csv

_explainer_instance = None
_predictors = {
        "zero_shot" : ZeroShotPredictor(config),
        "fine_tuned" : FineTunedPredictor(config)
    }

def _get_explainer() -> Explainer:
    global _explainer_instance
    if _explainer_instance is None:
        _explainer_instance = Explainer(config)

    return _explainer_instance

def explain_tool(input : ExplanationInput):

    _explainer_instance = _get_explainer()
    model = _predictors[input.chosen_model]
    points,labels = load_from_user_csv(input.path)
    predictions,explanations,metrics = _explainer_instance.explain(model,points,labels)
    precision,recall,f1_score = metrics

    return {
        "predictions" : predictions,
        "explanations" : explanations,
        "precision" : precision,
        "recall" : recall,
        "f1-score": f1_score
    }