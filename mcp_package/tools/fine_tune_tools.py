from truth_predictor.fine_tuned_predictor import FineTunedPredictor
from truth_predictor.utils import store_endpoint
from mcp_package.schemas.schemas import SimplePathInput
from mcp_package.config import config

_fine_tune_instance = None

def _get_fine_tune_instance() -> FineTunedPredictor:
    global _fine_tune_instance
    if _fine_tune_instance is None:
        _fine_tune_instance = FineTunedPredictor(config)

    return _fine_tune_instance

def fine_tune_tool(input: SimplePathInput):

    _fine_tune_instance = _get_fine_tune_instance()

    fine_tuning_endpoint = _fine_tune_instance.fine_tune(input.path)
    store_endpoint(fine_tuning_endpoint,config.output_path)
    return {"endpoint" : fine_tuning_endpoint}