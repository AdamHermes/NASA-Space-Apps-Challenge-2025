from fastapi import APIRouter, HTTPException
from pathlib import Path
import pandas as pd
import httpx
import json 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from typing import List, Tuple, Dict, Any, Optional
from tqdm import tqdm
from app.routers.visualization import find_by_kepid
import zipfile
from fastapi.responses import StreamingResponse
import io




router = APIRouter(prefix='/light_curve', tags=['light_curve'])

TCE_DIR = Path("app/storage/tces")
TCE_DIR.mkdir(parents=True, exist_ok=True)
CUMULATIVE_PATH = Path("app/storage/cumulative_new.csv")

TCE_MP4_DIR = Path("app/videos")
TCE_MP4_DIR.mkdir(parents=True, exist_ok=True)


BASE_URL = "https://exo.mast.stsci.edu/api/v0.1/dvdata"

cumulative_df = pd.read_csv(CUMULATIVE_PATH)


@router.get("/get_video_light_curve/{kepler_id}")
async def get_video_light_curve(kepler_id: str):
    """
    Handle light curve video generation for a Kepler system.
    Steps:
      1. Check existing MP4s; if all exist → return immediately.
      2. Ensure all TCE JSONs are present (fetch if missing).
      3. Generate missing videos via animate_cycle_reveal.
      4. Return summary of results.
    """
    
    # -----------------------------
    # Step 1: Check existing videos
    # -----------------------------
    existing_videos = await check_video_exists(kepler_id)
    if all(v["exists"] for v in existing_videos):
        return {
            "message": "Generate videos succesfully",
            "success": True
        }

    
    # -----------------------------
    # Step 2: Ensure TCE JSONs exist
    # -----------------------------
    json_status = await check_tce_json_exists(kepler_id)
    missing_jsons = [j for j in json_status if not j["exists"]]

    if missing_jsons:
        print(f"⚙️ Missing {len(missing_jsons)} TCE JSONs — fetching now...")
        await curl_reformat_tces(kepler_id)
        # Recheck JSONs after fetching
        json_status = await check_tce_json_exists(kepler_id)

    # Map JSON info by kepler_name for easy lookup
    json_map = {Path(j["json_path"]).stem.split("_", 1)[1]: j for j in json_status}

    
    # -----------------------------
    # Step 3: Generate missing videos
    # -----------------------------
    generated_videos = []
    for v in existing_videos:
        if v["exists"]:
            generated_videos.append(v)
            continue  # Skip existing ones

        kepler_name = v["kepler_name"]
        safe_name = kepler_name.replace(" ", "_").replace("-", "_")

        json_path = TCE_DIR / f"{kepler_id}_{safe_name}.json"
        video_path = TCE_MP4_DIR / f"{kepler_id}_{safe_name}.mp4"

        if not json_path.exists():
            print(f"⚠️ JSON missing for {kepler_name}, skipping video generation.")
            v["exists"] = False
            v["video_path"] = str(video_path)
            generated_videos.append(v)
            continue

        print(f"🎬 Generating video for {kepler_name}...")
        try:
            _, out_written = animate_cycle_reveal(
                json_path=str(json_path),
                out_path=str(video_path),
                seconds_per_cycle=5.0,
                fps=20,
                bitrate=2500,
                show=False,
                title=f"Light Curve — {kepler_name}",
            )
            v["exists"] = out_written is not None
            v["video_path"] = str(video_path)
            generated_videos.append(v)
        except Exception as e:
            print(f"❌ Failed to generate video for {kepler_name}: {e}")
            v["exists"] = False
            v["video_path"] = str(video_path)
            generated_videos.append(v)

    # -----------------------------
    # Step 4: Return response
    # -----------------------------
    all_done = all(gv["exists"] for gv in generated_videos)
    status = "completed" if all_done else "partial"
    message = (
        f"All videos successfully generated for Kepler {kepler_id}."
        if all_done
        else f"Some videos failed or missing for Kepler {kepler_id}."
    )

    return {
            "message": "Generate videos succesfully",
            "success": True
        }

    return {
        "kepler_id": kepler_id,
        "status": status,
        "message": message,
        "number_videos": len(generated_videos),
        "videos": generated_videos
    }
   
async def check_tce_json_exists(kepler_id: str) -> List[Dict[str, Any]]:
    """
    Check whether TCE JSON files already exist for a given Kepler system.

    - Looks up all TCEs and associated Kepler names via `find_by_hostname`.
    - Returns a list of dicts describing JSON availability for each TCE.
    - Each entry: {"kepler_name": str, "json_path": str, "exists": bool}
    """
    results = []
    res = await find_by_kepid(int(kepler_id))

    if not res or "data" not in res:
        raise HTTPException(status_code=404, detail=f"No entries found for Kepler ID {kepler_id}")

    for item in res["data"]:
        kepler_name = item.get("kepler_name", "Unknown") or "Unknown"
        safe_name = kepler_name.replace(" ", "_").replace("-", "_")
        json_path = TCE_DIR / f"{kepler_id}_{safe_name}.json"
        exists = json_path.exists()

        results.append({
            "kepler_name": kepler_name,
            "json_path": str(json_path),
            "exists": exists,
        })

    return results


async def check_video_exists(kepler_id: str) -> List[Dict[str, Any]]:
    """
    Check whether animation videos already exist for a given Kepler system.

    - Looks up all TCEs and their associated Kepler names using `find_by_hostname`.
    - Returns a list of dicts, each describing whether the MP4 file for that planet exists.
    - Each entry: {"kepler_name": str, "video_path": str, "exists": bool}
    """

    results = []
    res = await find_by_kepid(int(kepler_id))

    if not res or "data" not in res:
        raise HTTPException(status_code=404, detail=f"No entries found for Kepler ID {kepler_id}")

    for item in res["data"]:
        kepler_name = item.get("kepler_name", "Unknown") or "Unknown"

        safe_name = kepler_name.replace(" ", "_").replace("-", "_")
        video_path = TCE_MP4_DIR / f"{kepler_id}_{safe_name}.mp4"

        exists = video_path.exists()
        results.append({
            "kepler_name": kepler_name,
            "video_path": str(video_path),
            "exists": exists,
        })

    return results


@router.get("/curl_reformat_tces/{kepler_id}")
async def curl_reformat_tces(kepler_id: str):
    """
    Fetch ALL available light curves for every TCE associated with a given Kepler ID.
    Saves each TCE as its own JSON file (with Kepler name in filename if available).
    Returns a combined dict.
    """
    # Step 1: Fetch list of TCEs
    tce_url = f"{BASE_URL}/kepler/{kepler_id}/tces/"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(tce_url)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"TCE request failed: {resp.text}")

    try:
        tce_json = resp.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON from TCE endpoint")

    # Extract TCE list
    tces = tce_json["TCE"]
    if not tces:
        raise HTTPException(status_code=404, detail=f"No TCEs found for Kepler ID {kepler_id}")

    # Step 1.5: Build a lookup from CSV (map tce → kepler_name)
    mappings = {}
    subset = cumulative_df[cumulative_df["kepid"] == int(kepler_id)]
    for _, row in subset.iterrows():
        # kepoi_name looks like K00752.01 → tce = "1"
        tce_num = row["kepoi_name"].split(".")[1].lstrip("0")  # "01" → "1"
        kepler_name = row["kepler_name"]
        if pd.isna(kepler_name) or str(kepler_name).strip() == "":
            kepler_name = "Unknown"
        mappings[tce_num] = kepler_name

    print(mappings)

    # Step 2: Fetch data for all TCEs
    all_light_curves = {}
    async with httpx.AsyncClient(timeout=60.0) as client:
        for tce_id in tces:
            table_url = f"{BASE_URL}/kepler/{kepler_id}/table?tce={tce_id}"
            r = await client.get(table_url)
            if r.status_code != 200:
                print(f"⚠️ Skipped TCE {tce_id}: {r.status_code}")
                continue

            try:
                data_json = r.json()
            except json.JSONDecodeError:
                print(f"⚠️ Invalid JSON for TCE {tce_id}")
                continue

            df = pd.DataFrame(data_json.get("data", []))
            if df.empty:
                continue

            selected = [c for c in ["PHASE", "LC_DETREND", "MODEL_INIT"] if c in df.columns]
            df = df[selected].dropna(subset=["LC_DETREND"])

            # Pick friendly name if exists
            processed = process_tce_lightcurve(df, max_cycles=5)
            index_tce = (tce_id.split("_")[1])
            planet_name = mappings[index_tce]
            
            tce_result = {
                "kepler_id": kepler_id,
                "tce": tce_id,
                "name": planet_name,
                "fields": selected,
                "n_points": len(df),
                "cycles": processed["cycles"],  
                #"data": df.to_dict(orient="records"),
            }

            # Save each TCE separately
            safe_name = planet_name.replace(" ", "_").replace("-","_") 
            tce_out_path = TCE_DIR / f"{kepler_id}_{safe_name}.json"
            with open(tce_out_path, "w") as f:
                json.dump(tce_result, f, indent=2)

            all_light_curves[tce_id] = tce_result

    # Step 3: Save combined
    combined_result = {"kepler_id": kepler_id, "tces": tces, "light_curves": all_light_curves}


    return combined_result

def process_tce_lightcurve(df: pd.DataFrame, max_cycles: int = 5):
    """
    Process a single TCE light curve dataframe:
    - Detect phase cycle boundaries
    - Keep only up to `max_cycles`
    - If fewer cycles are available, automatically augment them by repeating
      the last available cycles to reach the desired count (for animation consistency)

    Returns
    -------
    dict with {"cycles": [[{PHASE, LC_DETREND, MODEL_INIT}, ...], ...]}
    """

    if df.empty or "PHASE" not in df.columns:
        return {"cycles": []}

    # Sort by index to keep temporal order
    df = df.sort_index().reset_index(drop=True)
    x = df["PHASE"].to_numpy()

    # Step 1: Drop leading rows until first negative PHASE (beginning of first full orbit)
    if np.any(x < 0):
        start_idx = np.argmax(x < 0)
        df = df.iloc[start_idx:].reset_index(drop=True)

    # Step 2: Detect cycle boundaries (PHASE decreases -> new orbit)
    break_indices = [0]
    for i in range(1, len(df)):
        if df["PHASE"].iloc[i] < df["PHASE"].iloc[i - 1]:
            break_indices.append(i)
    break_indices.append(len(df))

    # Step 3: Slice cycles
    raw_cycles = []
    for j in range(len(break_indices) - 1):
        start, end = break_indices[j], break_indices[j + 1]
        segment = df.iloc[start:end]
        if not segment.empty:
            raw_cycles.append(segment.to_dict(orient="records"))

    # Step 4: Auto-augment if not enough cycles
    if len(raw_cycles) == 0:
        return {"cycles": []}

    cycles = raw_cycles.copy()
    while len(cycles) < max_cycles:
        # Duplicate last few cycles to pad up to max_cycles
        needed = max_cycles - len(cycles)
        extension = raw_cycles[-min(len(raw_cycles), needed):]
        # Deep-copy to avoid modifying originals
        cycles.extend([json.loads(json.dumps(c)) for c in extension])

    # Step 5: Trim down if overshoot
    cycles = cycles[:max_cycles]

    return {"cycles": cycles}



def animate_cycle_reveal(
    json_path: str,
    out_path: str = "output.mp4",
    *,
    seconds_per_cycle: float = 5.0,
    fps: int = 30,
    dense_n: int = 800,
    bitrate: int = 2500,
    title: str = "Exoplanet Light Curve — Smooth left→right reveal per cycle",
    obs_label: str = "LC_DETREND (Observed)",
    model_label: str = "MODEL_INIT (Model)",
    obs_color: str = "blue",
    model_color: str = "red",
    show: bool = False,
) -> Tuple[FuncAnimation, Optional[str]]:
    """
    Build a smooth left→right reveal animation for each cycle in a cycle-based JSON file.

    Parameters
    ----------
    json_path : str
        Path to the input JSON with structure: {"cycles": [ [ {PHASE, LC_DETREND, MODEL_INIT}, ... ], ... ]}.
    out_path : str
        Output video file path (.mp4). Requires ffmpeg available on PATH.
    seconds_per_cycle : float
        Duration to reveal each cycle.
    fps : int
        Frames per second.
    dense_n : int
        Number of points in the dense interpolation grid per cycle.
    bitrate : int
        Bitrate for the FFMpegWriter.
    title : str
        Plot title.
    obs_label : str
        Legend label for observed curve.
    model_label : str
        Legend label for model curve.
    obs_color : str
        Matplotlib color for observed curve.
    model_color : str
        Matplotlib color for model curve.
    show : bool
        If True, show the figure at the end (blocks).

    Returns
    -------
    (ani, out_path_or_None)
        ani: the FuncAnimation object (useful for embedding in notebooks).
        out_path_or_None: the output path if saved, else None.

    Raises
    ------
    ValueError
        If no cycles are found.
    """

    # -----------------------------
    # Load cycle-based file
    # -----------------------------
    with open(json_path, "r") as f:
        raw = json.load(f)

    cycles: List[List[Dict[str, Any]]] = raw.get("cycles", [])
    n_cycles = len(cycles)
    if n_cycles == 0:
        raise ValueError("No cycles found in file.")

    # -----------------------------
    # Helpers
    # -----------------------------
    def build_dense_pair(cycle: List[Dict[str, Any]], dense_n_local: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns: (x_dense, y_obs_dense, y_mod_dense)
        - If a curve lacks >=2 finite points, its dense version is all-NaN
        - x_dense spans the finite PHASE range in this cycle
        """
        x = np.array([row.get("PHASE", np.nan) for row in cycle], dtype=float)
        y_obs = np.array([row.get("LC_DETREND", np.nan) for row in cycle], dtype=float)
        y_mod = np.array([row.get("MODEL_INIT", np.nan) for row in cycle], dtype=float)

        m = np.isfinite(x)
        x, y_obs, y_mod = x[m], y_obs[m], y_mod[m]
        if x.size < 2:
            return x, np.full_like(x, np.nan), np.full_like(x, np.nan)

        order = np.argsort(x)
        x, y_obs, y_mod = x[order], y_obs[order], y_mod[order]

        x_dense = np.linspace(x.min(), x.max(), dense_n_local)

        def safe_interp(y_src: np.ndarray) -> np.ndarray:
            ok = np.isfinite(y_src)
            if ok.sum() >= 2:
                return np.interp(x_dense, x[ok], y_src[ok])
            else:
                return np.full_like(x_dense, np.nan, dtype=float)

        y_obs_dense = safe_interp(y_obs)
        y_mod_dense = safe_interp(y_mod)
        return x_dense, y_obs_dense, y_mod_dense

    def prefix_xy(x_dense: np.ndarray, y_dense: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return first k points, dropping NaNs in that slice so the line doesn't break."""
        xk = x_dense[:k]
        yk = y_dense[:k]
        m = np.isfinite(xk) & np.isfinite(yk)
        return xk[m], yk[m]

    # -----------------------------
    # Precompute dense pairs
    # -----------------------------
    dense_pairs = [build_dense_pair(c, dense_n) for c in cycles]

    # -----------------------------
    # Axis limits (robust to NaNs)
    # -----------------------------
    all_x = np.concatenate([p[0] for p in dense_pairs if p[0].size > 0] or [np.array([0.0, 1.0])])
    all_y_candidates = []
    for xd, yo, ym in dense_pairs:
        if xd.size:
            if np.isfinite(yo).any():
                all_y_candidates.append(yo[np.isfinite(yo)])
            if np.isfinite(ym).any():
                all_y_candidates.append(ym[np.isfinite(ym)])
    if len(all_y_candidates) == 0:
        ymin, ymax = -1.0, 1.0
    else:
        all_y = np.concatenate(all_y_candidates)
        ymin, ymax = float(np.nanmin(all_y)), float(np.nanmax(all_y))
        if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin == ymax:
            ymin, ymax = -1.0, 1.0
    pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)

    # -----------------------------
    # Timing
    # -----------------------------
    frames_per_cycle = int(round(seconds_per_cycle * fps))
    total_frames = n_cycles * frames_per_cycle

    # -----------------------------
    # Figure & artists
    # -----------------------------
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(title)
    ax.set_xlabel("Orbital Phase")
    ax.set_ylabel("Flux (Detrended)")
    ax.set_xlim(float(np.min(all_x)), float(np.max(all_x)))
    ax.set_ylim(ymin - pad, ymax + pad)

    (line_obs,) = ax.plot([], [], lw=2, color=obs_color, label=obs_label)
    (line_mod,) = ax.plot([], [], lw=2, color=model_color, label=model_label)
    cycle_txt = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top", ha="left")
    ax.legend(loc="best")

    # -----------------------------
    # Animation update
    # -----------------------------
    def update(frame: int):
        cycle_idx = frame // frames_per_cycle
        step_in_cycle = frame % frames_per_cycle
        if cycle_idx >= n_cycles:
            return line_obs, line_mod, cycle_txt

        x_dense, y_obs_dense, y_mod_dense = dense_pairs[cycle_idx]
        if x_dense.size < 2:
            line_obs.set_data([], [])
            line_mod.set_data([], [])
            cycle_txt.set_text(f"Cycle {cycle_idx+1}/{n_cycles}")
            return line_obs, line_mod, cycle_txt

        frac = step_in_cycle / max(1, (frames_per_cycle - 1))
        n = x_dense.size
        k = max(2, 1 + int(frac * (n - 1)))

        xo, yo = prefix_xy(x_dense, y_obs_dense, k)
        line_obs.set_data(xo, yo)

        xm, ym = prefix_xy(x_dense, y_mod_dense, k)
        line_mod.set_data(xm, ym)

        cycle_txt.set_text(f"Cycle {cycle_idx+1}/{n_cycles}")
        return line_obs, line_mod, cycle_txt

    ani = FuncAnimation(fig, update, frames=total_frames, interval=1000 / fps, blit=False)

    # -----------------------------
    # Save (mp4 via ffmpeg)
    # -----------------------------
    out_written: Optional[str] = None
    try:
        writer = FFMpegWriter(fps=fps, bitrate=bitrate)
        pbar = tqdm(total=total_frames, desc="Saving video", unit="frame")

        def progress_callback(i, n):
            # Matplotlib calls this with i from 0..n-1
            pbar.update(1)

        ani.save(out_path, writer=writer, progress_callback=progress_callback)
        pbar.close()
        out_written = out_path
    except FileNotFoundError as e:
        # ffmpeg not found
        print("ERROR: ffmpeg not found on PATH. Install it (e.g., `brew install ffmpeg`) to save MP4.")
        out_written = None

    if show:
        plt.show()
    else:
        plt.close(fig)

    return ani, out_written


def zip_files_in_memory(file_paths: list[str], zip_name: str = "videos.zip") -> io.BytesIO:
    """
    Create an in-memory ZIP archive containing the given file paths.
    Returns a BytesIO object ready to be streamed or saved.
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for path in file_paths:
            file_path = Path(path)
            if file_path.exists():
                zipf.write(file_path, arcname=file_path.name)
            else:
                print(f"⚠️ File not found: {file_path}")
    zip_buffer.seek(0)
    return zip_buffer