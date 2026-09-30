from .config import Config
from google import genai
from .utils import BatchResponseSchema, PredictSchema
from sklearn.metrics import f1_score,precision_score,recall_score
from google.genai.types import GenerateContentConfig, Part
import json

class ZeroShotPredictor:

    def __init__(self,config: Config):
        self.model_name = config.base_llm
        self.project_id = config.project_id
        self.location = config.location
        self.client = genai.Client(vertexai=True,project=self.project_id,location=self.location)
        self.response_config = GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                    seed=42,
                    top_k=1,
                    top_p=1.0,
                    response_schema=BatchResponseSchema[PredictSchema]
                )

    def predict(self,points,labels=None):
        
        responses = self.send_request(points)

        predictions = [item["prediction"] for item in responses.evaluations]
        
        metrics = None    
        if labels is not None: 
            metrics = self.calculate_metrics(y_true=labels,y_pred=predictions)

        return predictions,metrics

    def calculate_metrics(self,y_true,y_pred):

        precision = precision_score(y_true,y_pred,average='weighted') #this is a priority due to FPs being more dangerous: False statements being clf as "True"
        recall = recall_score(y_true,y_pred,average='weighted')
        f1 = f1_score(y_true,y_pred,average='weighted')

        return precision,recall,f1

    def send_request(self,points):

        response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[
                        Part.from_text(text=f"Evaluate each of the {len(points)} statements below. Return exactly that many evaluations, in the same order."),
                        Part.from_text(text=json.dumps(points)),
                    ],
                    config= self.response_config
                )
        
        responses = BatchResponseSchema.model_validate_json(response.text)

        return responses