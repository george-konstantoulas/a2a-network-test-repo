from truth_predictor.zero_shot_predictor import ZeroShotPredictor
from truth_predictor.fine_tuned_predictor import FineTunedPredictor
from mcp_package.schemas.schemas import SimplePathInput
from mcp_package.config import config
from truth_predictor.utils import load_from_user_csv

_zero_shot_predictor_instance = None
_fine_tuned_predictor_instance = None

def _get_zero_shot_predictor() -> ZeroShotPredictor:
    global _zero_shot_predictor_instance
    if _zero_shot_predictor_instance is None:
        _zero_shot_predictor_instance = ZeroShotPredictor(config)

    return _zero_shot_predictor_instance

def _get_fine_tuned_predictor() -> FineTunedPredictor:
    global _fine_tuned_predictor_instance
    if _fine_tuned_predictor_instance is None:
        _fine_tuned_predictor_instance = FineTunedPredictor(config)

    return _fine_tuned_predictor_instance

def zero_shot_predict_tool(input : SimplePathInput):

    _zero_shot_predictor_instance = _get_zero_shot_predictor()
    points,labels = load_from_user_csv(input.path)
    predictions,metrics = _zero_shot_predictor_instance.predict(points,labels)
    precision,recall,f1_score = metrics
        
    return {
        "predictions" : predictions,
        "precision" : precision,
        "recall" : recall,
        "f1-score": f1_score
    }

def fine_tuned_predict_tool(input: SimplePathInput):

    _fine_tuned_predictor_instance = _get_fine_tuned_predictor()
    points,labels = load_from_user_csv(input.path)
    predictions,metrics = _fine_tuned_predictor_instance.predict(points,labels)
    precision,recall,f1_score = metrics
    
    return {
        "predictions" : predictions,
        "precision" : precision,
        "recall" : recall,
        "f1-score": f1_score
    }