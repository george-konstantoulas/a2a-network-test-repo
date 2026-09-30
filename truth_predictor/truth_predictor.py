from .config import Config
from .explainer import Explainer
from .zero_shot_predictor import ZeroShotPredictor
from .fine_tuned_predictor import FineTunedPredictor
from .utils import *

class TruthPredictor:

    def __init__(self,config: Config):
        self.zero_shot_predictor = ZeroShotPredictor(config)
        self.sft_predictor = FineTunedPredictor(config)
        self.explainer = Explainer(config)

    def predict(self,points,labels=None):
        
        return self.zero_shot_predictor.predict(points,labels)

    def fine_tune(self,training_dataset):

        return self.sft_predictor.fine_tune(training_dataset)

    def save_model_endpoint(self,endpoint):

        store_endpoint(endpoint,self.sft_predictor.output_path)

        return

    def explain(self,model,points,labels=None):

        predictions,explanations,metrics = self.explainer.explain(model,points,labels)

        return predictions,explanations,metrics