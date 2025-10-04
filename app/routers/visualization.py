from fastapi import APIRouter, HTTPException
from pathlib import Path
import pandas as pd 

router = APIRouter(prefix='/visualization', tags=['visualization'])

CSV_DIR = Path("storage/uploaded_csvs")

@router.get("/find_by_hostname/{hostname}")
async def find_by_hostname(hostname: str):
    """
    Find all rows with matching kepid based on the given hostname.
    """
    try:
        csv_file_path = CSV_DIR / "cumulative_with_hostnames.csv"
        if not csv_file_path.exists():
            raise HTTPException(status_code=404, detail="CSV file not found")
        df = pd.read_csv(csv_file_path)
        if 'Host Name' not in df.columns or 'kepid' not in df.columns:
            raise HTTPException(status_code=400, detail="Required columns 'Host Name' or 'kepid' not found in CSV")
        matching_rows = df[df['Host Name'].str.contains(hostname, case=False, na=False)]
        if matching_rows.empty:
            raise HTTPException(status_code=404, detail=f"No rows found with hostname containing '{hostname}'")
        kepid = matching_rows.iloc[0]['kepid']
        all_matching_kepid_rows = df[df['kepid'] == kepid]
        result = all_matching_kepid_rows.to_dict('records')
        
        # Clean the data to handle non-JSON compliant values (NaN, inf, -inf)
        cleaned_result = []
        for row in result:
            cleaned_row = {}
            for key, value in row.items():
                if pd.isna(value):
                    cleaned_row[key] = None
                elif isinstance(value, float):
                    if value == float('inf'):
                        cleaned_row[key] = None
                    elif value == float('-inf'):
                        cleaned_row[key] = None
                    else:
                        cleaned_row[key] = value
                else:
                    cleaned_row[key] = value
            cleaned_result.append(cleaned_row)
        
        return {
            "hostname": hostname,
            "kepid": int(kepid),
            "total_rows": len(cleaned_result),
            "data": cleaned_result
        }
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="CSV file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
