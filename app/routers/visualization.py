from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import pandas as pd
import httpx
import json
import os 

router = APIRouter(prefix='/visualization', tags=['visualization'])

CSV_DIR = Path("app/storage/")
VIDEO_DIR = Path("app/videos/")

@router.get("/find_by_hostname/{hostname}")
async def find_by_hostname(hostname: str):
    """
    Find all distinct hostnames matching the search pattern.
    """
    try:
        csv_file_path = CSV_DIR / "cumulative_with_hostnames.csv"
        if not csv_file_path.exists():
            raise HTTPException(status_code=404, detail="CSV file not found")
        df = pd.read_csv(csv_file_path)
        if 'Host Name' not in df.columns:
            raise HTTPException(status_code=400, detail="Required column 'Host Name' not found in CSV")
        
        # Find all rows matching the hostname pattern
        matching_rows = df[df['Host Name'].str.contains(hostname, case=False, na=False)]
        if matching_rows.empty:
            raise HTTPException(status_code=404, detail=f"No hostnames found containing '{hostname}'")
        
        # Get unique hostnames and their planet counts
        hostname_groups = matching_rows.groupby('Host Name').agg({
            'kepid': 'first',  # Get the kepid for navigation
            'Host Name': 'count'  # Count planets per hostname
        }).rename(columns={'Host Name': 'planet_count'}).reset_index()
        
        # Convert to list of dictionaries
        result = []
        for _, row in hostname_groups.iterrows():
            result.append({
                'hostname': row['Host Name'],
                'kepid': int(row['kepid']),
                'planet_count': int(row['planet_count'])
            })
        
        return {
            "search_term": hostname,
            "total_hostnames": len(result),
            "hostnames": result
        }
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="CSV file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@router.get("/find_by_kepid/{kepid}")
async def find_by_kepid(kepid: int):
    """
    Find all rows with matching kepid.
    """
    try:
        csv_file_path = CSV_DIR / "cumulative_with_hostnames.csv"
        if not csv_file_path.exists():
            raise HTTPException(status_code=404, detail="CSV file not found")
        df = pd.read_csv(csv_file_path)
        if 'kepid' not in df.columns:
            raise HTTPException(status_code=400, detail="Required column 'kepid' not found in CSV")
        matching_rows = df[df['kepid'] == kepid]
        if matching_rows.empty:
            raise HTTPException(status_code=404, detail=f"No rows found with kepid '{kepid}'")
        
        result = matching_rows.to_dict('records')
        
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
            "kepid": kepid,
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
    
@router.get("/hostids")
async def get_all_hostid():
    "Get all existing host ids from the CSV file where host name is not empty."
    try:
        csv_file_path = CSV_DIR / "cumulative_with_hostnames.csv"
        if not csv_file_path.exists():
            raise HTTPException(status_code=404, detail="CSV file not found")
        
        df = pd.read_csv(csv_file_path)
        if 'Host Name' not in df.columns or 'kepid' not in df.columns:
            raise HTTPException(status_code=400, detail="Required columns 'Host Name' and 'kepid' not found in CSV")
        filtered_df = df[df['Host Name'].notna() & (df['Host Name'] != '')]
        hostids = filtered_df['kepid'].unique().tolist()
        
        return {
            "total_ids": len(hostids),
            "hostids": sorted(hostids)
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


@router.get("/videos")
async def get_videos():
    """
    Get list of available video files.
    """
    try:
        if not VIDEO_DIR.exists():
            VIDEO_DIR.mkdir(parents=True, exist_ok=True)
            return {
                "total_videos": 0,
                "videos": []
            }
        
        video_files = []
        for file_path in VIDEO_DIR.glob("*.mp4"):
            video_files.append({
                "filename": file_path.name,
                "title": file_path.stem.replace("_", " ").title(),
                "size": file_path.stat().st_size
            })
        
        return {
            "total_videos": len(video_files),
            "videos": sorted(video_files, key=lambda x: x["title"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting videos: {str(e)}")


@router.get("/videos/{filename}")
async def get_video(filename: str):
    """
    Serve a video file by filename.
    """
    try:
        if not VIDEO_DIR.exists():
            raise HTTPException(status_code=404, detail="Video directory not found")
        
        video_path = VIDEO_DIR / filename
        
        if not video_path.exists():
            raise HTTPException(status_code=404, detail=f"Video file '{filename}' not found")
        
        if not video_path.suffix.lower() == '.mp4':
            raise HTTPException(status_code=400, detail="Only MP4 files are supported")
        
        return FileResponse(
            path=str(video_path),
            media_type='video/mp4',
            filename=filename
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving video: {str(e)}")

