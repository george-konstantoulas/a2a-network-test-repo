from .config import Config
from .utils import *
from .zero_shot_predictor import ZeroShotPredictor
from .fine_tuned_predictor import FineTunedPredictor
from google.genai.types import GenerateContentConfig

class Explainer(ZeroShotPredictor):

    def __init__(self,config: Config):
        super().__init__(config)
        self.response_config = GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.0,
                            seed=42,
                            top_k=1,
                            top_p=1.0,
                            response_schema=BatchResponseSchema[ExplainerSchema]
                        )

    def explain(self, model: ZeroShotPredictor | FineTunedPredictor,points,labels=None):

        self.model_name = model.model_name
        responses = self.send_request(points)

        predictions = [item["prediction"] for item in responses.evaluations]
        explanations = [item["explanation"] for item in responses.evaluations]

        metrics = None
        if labels is not None:
            metrics = self.calculate_metrics(y_true=labels,y_pred=predictions)

        return predictions,explanations,metrics