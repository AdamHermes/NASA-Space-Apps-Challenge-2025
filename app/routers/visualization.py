from fastapi import APIRouter, HTTPException
from pathlib import Path
import pandas as pd
import httpx
import json 

router = APIRouter(prefix='/visualization', tags=['visualization'])

CSV_DIR = Path("app/uploaded_csvs")

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


@router.get("/hostnames")
async def get_all_hostnames():
    "Get all existing host names from the CSV file."
    try:
        csv_file_path = CSV_DIR / "cumulative_with_hostnames.csv"
        if not csv_file_path.exists():
            raise HTTPException(status_code=404, detail="CSV file not found")
        
        df = pd.read_csv(csv_file_path)
        if 'Host Name' not in df.columns:
            raise HTTPException(status_code=400, detail="Required column 'Host Name' not found in CSV")
        
        # Get unique host names, removing NaN values
        hostnames = df['Host Name'].dropna().unique().tolist()
        
        return {
            "total_hostnames": len(hostnames),
            "hostnames": sorted(hostnames)
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="CSV file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@router.get("/lightcurve/{star_id}/{tce_num}")
async def get_lightcurve(star_id: str, tce_num: str):
    """
    Fetch lightcurve data from MAST API for the given star_id and tce number.
    """
    try:
        url = f"https://exo.mast.stsci.edu/api/v0.1/dvdata/kepler/{star_id}/table?tce={tce_num}"
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            if response.status_code == 404:
                raise HTTPException(status_code=404, detail=f"Lightcurve data not found for star_id: {star_id}")
            elif response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code, 
                    detail=f"Error fetching lightcurve data: {response.text}"
                )
            
            data = response.json()
            return {
                "star_id": star_id,
                "data": data
            }
            
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Error parsing response: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")