from pandas import read_csv
from mcp_package.schemas.schemas import SimplePathInput
PREVIEW_ROWS = 3
PREVIEW_CHARS = 200

def dataset_info_tool(input_path: SimplePathInput):
    """
    Summarise useful info and check dataset validity.
    """
    df = read_csv(input_path.path)

    if "statement" not in df.columns:
        raise ValueError(f"No 'statement' column in {input_path.path}. Found {list(df.columns)}")

    labels_provided = "label" in df.columns

    return {
        "row_count": int(len(df)),
        "columns": df.columns.tolist(),
        "labels_provided": labels_provided,
        "label_distribution": (
            df["label"].astype(str).value_counts().to_dict() if labels_provided else None,
        ),
        "preview": [s[:PREVIEW_CHARS] for s in df["statement"].head(PREVIEW_ROWS)]
    }