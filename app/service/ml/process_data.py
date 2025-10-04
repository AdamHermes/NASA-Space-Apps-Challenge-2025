import pandas as pd
from io import BytesIO
from split import split


def process_data(file_bytes: bytes):
    try:
        df = pd.read_csv(BytesIO(file_bytes))
        data = split(df)
        return data

    except Exception as e:
        return {
            "error": f"Failed to process CSV: {str(e)}",
            "status": "500"
        }
