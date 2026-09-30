from dataclasses import dataclass
from google.genai.types import GenerateContentConfig

@dataclass
class Config():
    base_llm = "gemini-3.1-flash-lite"
    response_config = GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.0,
        seed=42,
        top_k=1,
        top_p=1.0
    )
    output_path : str
    project_id : str
    bucket_name : str
    location : str