import pandas as pd
from sklearn.model_selection import train_test_split
import fsspec
import json
from google.cloud import storage
from os.path import basename
from vertexai.tuning import sft
import time
from pydantic import BaseModel, Field
from typing import Literal, List, Generic, TypeVar
from sys import stderr

T = TypeVar("T")

class PredictSchema(BaseModel):
    prediction: Literal["True","False"] = Field(
        description="Assesement of truthfulness of a statement."
    )

    
class ExplainerSchema(BaseModel):
    explanation: str = Field(
        description="""
                    Step-by-step evaluation of row data
                    to determine statement truthfulness.
                    """
    )
    prediction: Literal["True","False"] = Field(
        description="Final decision based strictly on the reasoning."
    )

class BatchResponseSchema(BaseModel, Generic[T]):
    evaluations : List[T] = Field(
        description="List of predictions matching every statement in the provided input."
    )

def load_from_user_csv(path : str):

    ground_labels = None
    statement_df = pd.read_csv(path)
    if "statement" not in statement_df.columns:
        raise ValueError("Statement not found in input.")

    if "label" in statement_df.columns:
        ground_labels = statement_df['label'].values.astype(str).tolist()
        statement_df = statement_df.drop(columns=['label'])

    return statement_df.to_csv(index=False), ground_labels

##### DATA PREPARATION FOR FINE TUNING #####
def split_data(training_dataset):

    data = pd.read_csv(training_dataset)

    train_set, temp_set = train_test_split(
        data,
        test_size=0.20,
        random_state=42,
        stratify=data['label']
    )

    val_set,test_set = train_test_split(
        temp_set,
        test_size=0.5,
        random_state=42,
        stratify=temp_set['label']
    )

    print(f"Train set size: {len(train_set)} samples\n",\
        f"Validation set size: {len(val_set)} samples\n",\
        f"Test set size: {len(test_set)} samples",file=stderr)

    return train_set,val_set,test_set

def convert_to_gemini_format(train_set : pd.DataFrame,val_set : pd.DataFrame,output_path):

    output_files = [f"{output_path}/train.jsonl",f"{output_path}/val.jsonl"]

    for df,file in zip([train_set,val_set],output_files):
        with open(file,"w",encoding="utf-8") as f:
            for _,row in df.iterrows():
                record = {}

                feature_cols = df.columns.values.tolist()
                feature_cols.remove("label")
                user_text = [f"{col}:\n{row[col]}" for col in feature_cols]
                user_prompt = "\n\n".join(user_text)

                model_response = json.dumps({"prediction": row['label']})

                record["contents"] = [
                    {
                        "role":"user",
                        "parts":[{"text":user_prompt}]
                    },
                    {
                        "role":"model",
                        "parts":[{"text":model_response}]
                    }
                ]

                f.write(json.dumps(record)+"\n")
            
    return output_files

def gcs_bucket_upload(local_files,bucket_name,project_id):

    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)

    for file in local_files:
        print(f"Uploading {file} to gs://{bucket_name}/tuning ...",file=stderr)
        blob = bucket.blob(f"tuning/{basename(file)}")
        blob.upload_from_filename(file)
        print("Upload complete.",file=stderr)

def store_test_set(test_set: pd.DataFrame,path):

    test_set.to_csv(f"{path}/test_set.csv",index=False)

    return

def train(model_name,blob_path):
    print("Launching SFT job...",file=stderr)

    sft_job = sft.train(
        source_model = model_name,
        train_dataset=f"gs://{blob_path}/train.jsonl",
        validation_dataset=f"gs://{blob_path}/val.jsonl",
        epochs=3,
        adapter_size=8,
        learning_rate_multiplier=1.0,
        tuned_model_display_name=f"{model_name}-truthful"
    )

    while not sft_job.has_ended:
        time.sleep(60)
        sft_job.refresh()
        print(f"Status: {sft_job.state.name}",file=stderr)

    print("Job Completed",file=stderr)
    print(f"Tuned Model Endpoint: {sft_job.tuned_model_endpoint_name}",file=stderr)

    return sft_job.tuned_model_endpoint_name

def store_endpoint(endpoint,output_path):

    with fsspec.open(f"{output_path}/endpoint.json","w") as f:

        json.dump({"endpoint":endpoint},f)

    return

