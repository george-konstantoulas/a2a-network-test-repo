from pydantic import BaseModel, Field
from typing import Literal

class SimplePathInput(BaseModel):
    path: str = Field(
        description="Path to dataset for parsing, summarization or fine tuning."
    )

class ExplanationInput(SimplePathInput):
    chosen_model : Literal["zero_shot","fine_tuned"] = Field(
        description="The predictor model to use for explanations."
    )