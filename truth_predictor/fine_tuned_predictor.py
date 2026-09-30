from .config import Config
from .zero_shot_predictor import ZeroShotPredictor
from .utils import *
from os import environ

class FineTunedPredictor(ZeroShotPredictor):

    def __init__(self,config : Config):
        super().__init__(config)
        self.output_path = config.output_path
        self.bucket_name = config.bucket_name
        self.model_name = environ.get("FINE_TUNED_ENDPOINT",config.base_llm)

    def fine_tune(self,training_dataset):

        train_set,val_set,test_set = split_data(training_dataset)
        store_test_set(test_set,self.output_path)
        output_paths = convert_to_gemini_format(train_set,val_set,self.output_path)
        gcs_bucket_upload(output_paths,self.bucket_name,self.project_id)

        ft_model_enpoint = train(self.model_name,f"{self.bucket_name}/tuning")

        return ft_model_enpoint