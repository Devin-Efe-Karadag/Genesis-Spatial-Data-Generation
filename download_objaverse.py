import glob
import gzip
import json
import multiprocessing
import numpy as np
import os
import urllib.request
import warnings
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm

BASE_PATH = os.path.join("./", "objaverse")
_VERSIONED_PATH = os.path.join(BASE_PATH, "hf-objaverse-v1")


def _load_object_paths() -> Dict[str, str]:
    """Load the object paths from the dataset.

    The object paths specify the location of where the object is located
    in the Hugging Face repo.

    Returns:
        A dictionary mapping the uid to the object path.
    """
    object_paths_file = "object-paths.json.gz"
    local_path = os.path.join(_VERSIONED_PATH, object_paths_file)

    if not os.path.exists(local_path):
        hf_url = f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/{object_paths_file}"
        # wget the file and put it in local_path
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        urllib.request.urlretrieve(hf_url, local_path)

    with gzip.open(local_path, "rb") as f:
        object_paths = json.load(f)
    return object_paths


def load_uids() -> List[str]:
    """Load the uids from the dataset.

    Returns:
        A list of uids.
    """
    return list(_load_object_paths().keys())


def _download_object(
    uid: str,
    object_path: str,
    total_downloads: float,
    start_file_count: int,
    download_folder: str,
) -> Tuple[str, str]:
    """Download the object for the given uid to the specified folder."""
    local_path = os.path.join(download_folder, object_path)
    tmp_local_path = local_path + ".tmp"
    hf_url = (
        f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/{object_path}"
    )
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    urllib.request.urlretrieve(hf_url, tmp_local_path)
    os.rename(tmp_local_path, local_path)

    files = glob.glob(os.path.join(download_folder, "glbs", "*", "*.glb"))
    print(
        "Downloaded",
        len(files) - start_file_count,
        "/",
        total_downloads,
        "objects",
    )

    return uid, local_path


def load_objects(
    uids: List[str], download_folder: str, download_processes: int = 1
) -> Dict[str, str]:
    """Return the path to the object files for the given uids."""
    object_paths = _load_object_paths()
    out = {}

    if download_processes == 1:
        uids_to_download = []
        for uid in uids:
            if uid.endswith(".glb"):
                uid = uid[:-4]
            if uid not in object_paths:
                warnings.warn(f"Could not find object with uid {uid}. Skipping it.")
                continue
            object_path = object_paths[uid]
            local_path = os.path.join(download_folder, object_path)
            if os.path.exists(local_path):
                out[uid] = local_path
                continue
            uids_to_download.append((uid, object_path))
        if len(uids_to_download) == 0:
            return out
        start_file_count = len(
            glob.glob(os.path.join(download_folder, "glbs", "*", "*.glb"))
        )
        for uid, object_path in uids_to_download:
            try:
                uid, local_path = _download_object(
                    uid,
                    object_path,
                    len(uids_to_download),
                    start_file_count,
                    download_folder,
                )
                out[uid] = local_path
            except Exception as e:
                print(f"{uid} failed (e)")
    else:
        args = []
        for uid in uids:
            if uid.endswith(".glb"):
                uid = uid[:-4]
            if uid not in object_paths:
                warnings.warn(f"Could not find object with uid {uid}. Skipping it.")
                continue
            object_path = object_paths[uid]
            local_path = os.path.join(download_folder, object_path)
            if not os.path.exists(local_path):
                args.append((uid, object_path))
            else:
                out[uid] = local_path
        if len(args) == 0:
            return out
        print(
            f"starting download of {len(args)} objects with {download_processes} processes"
        )
        start_file_count = len(
            glob.glob(os.path.join(download_folder, "glbs", "*", "*.glb"))
        )
        args_list = [
            (*arg, len(args), start_file_count, download_folder) for arg in args
        ]
        with multiprocessing.Pool(download_processes) as pool:
            r = pool.starmap(_download_object, args_list)
            for uid, local_path in r:
                out[uid] = local_path
    return out


def load_annotations(uids: Optional[List[str]] = None) -> Dict[str, Any]:
    """Load the full metadata of all objects in the dataset.

    Args:
        uids: A list of uids with which to load metadata. If None, it loads
        the metadata for all uids.

    Returns:
        A dictionary mapping the uid to the metadata.
    """
    metadata_path = os.path.join(_VERSIONED_PATH, "metadata")
    object_paths = _load_object_paths()
    dir_ids = (
        set(object_paths[uid].split("/")[1] for uid in uids)
        if uids is not None
        else [f"{i // 1000:03d}-{i % 1000:03d}" for i in range(160)]
    )
    if len(dir_ids) > 10:
        dir_ids = tqdm(dir_ids)
    out = {}
    for i_id in dir_ids:
        json_file = f"{i_id}.json.gz"
        local_path = os.path.join(metadata_path, json_file)
        if not os.path.exists(local_path):
            hf_url = f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/metadata/{i_id}.json.gz"
            # wget the file and put it in local_path
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            urllib.request.urlretrieve(hf_url, local_path)
        with gzip.open(local_path, "rb") as f:
            data = json.load(f)
        if uids is not None:
            data = {uid: data[uid] for uid in uids if uid in data}
        out.update(data)
        if uids is not None and len(out) == len(uids):
            break
    return out


with open("datalist_filtered_clip.txt", "r") as o:
    lines = o.readlines()

filtered_uids = list(map(lambda x: x.split(os.path.sep)[1], lines))
filtered_uids = np.unique(filtered_uids).tolist()

"""
uids = load_uids()
annotations = load_annotations(uids)
breakpoint()
selected_uids = [
    uid
    for uid, annotation in annotations.items()
    if (annotation["license"] == 'by') and (annotation['animationCount'] > 0) and (annotation["likeCount"] > 4)
]
load_objects(selected_uids, download_folder="filtered_objs", download_processes=1)
"""

load_objects(filtered_uids, download_folder="filtered_objs", download_processes=1)
