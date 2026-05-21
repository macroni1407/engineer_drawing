import os
import cv2
import time
import json
import numpy as np

from apscheduler.schedulers.background import (
    BackgroundScheduler
)

# ======================================================
# CACHE CONFIG
# ======================================================

CACHE_DIR = "./cache"

RESULT_DIR = os.path.join(
    CACHE_DIR,
    "results"
)

ZIP_DIR = os.path.join(
    CACHE_DIR,
    "zips"
)

GALLERY_DIR = os.path.join(
    CACHE_DIR,
    "gallery"
)

TTL = 300

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(ZIP_DIR, exist_ok=True)
os.makedirs(GALLERY_DIR, exist_ok=True)

# ======================================================
# SAVE RESULT CACHE
# ======================================================

def save_result_cache(
    image_hash,
    result,
):

    # IMPORTANT
    os.makedirs(
        RESULT_DIR,
        exist_ok=True
    )

    path = os.path.join(
        RESULT_DIR,
        f"{image_hash}.npz"
    )

    try:
        if os.path.exists(path):
            return path

        np.savez_compressed(
            path,
            result=result
        )
        return path

    except Exception as e:
        print(
            "Save cache failed:",
            e
        )
        return None


# ======================================================
# LOAD RESULT CACHE
# ======================================================
def load_result_cache(path):
    try:
        if not path:
            return None

        if not os.path.exists(path):
            return None

        data = np.load(
            path,
            allow_pickle=True
        )
        return data["result"]

    except Exception as e:
        print(
            "Load cache failed:",
            e
        )

        return None


# ======================================================
# SAVE GALLERY CACHE
# ======================================================
def save_gallery_cache(
    image_hash,
    crops,
):
    os.makedirs(
        GALLERY_DIR,
        exist_ok=True
    )

    gallery_folder = os.path.join(
        GALLERY_DIR,
        image_hash
    )

    os.makedirs(
        gallery_folder,
        exist_ok=True
    )

    gallery_items = []

    for idx, (
        img,
        label_name,
        score,
        obj_id
    ) in enumerate(crops):

        filename = (
            f"{idx}_{label_name}.png"
        )

        path = os.path.join(
            gallery_folder,
            filename
        )

        caption = (
            f"id: {obj_id} - {label_name}: {score}"
        )

        try:
            # already cached
            if not os.path.exists(path):
                img_bgr = cv2.cvtColor(
                    img,
                    cv2.COLOR_RGB2BGR
                )

                cv2.imwrite(
                    path,
                    img_bgr
                )

            gallery_items.append(
                (
                    path,
                    caption
                )
            )

        except Exception as e:
            print(
                "Save gallery failed:",
                e
            )

    return gallery_items


# ======================================================
# LOAD GALLERY CACHE
# ======================================================

def load_gallery_cache(
    gallery_items
):
    valid_items = []

    for path, caption in gallery_items:
        if os.path.exists(path):
            valid_items.append(
                (
                    path,
                    caption
                )
            )
    return valid_items


# ======================================================
# SAVE JSON CACHE
# ======================================================
def save_json_cache(
    path,
    data,
):
    try:
        with open(
            path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2
            )

    except Exception as e:
        print(
            "Save json failed:",
            e
        )


# ======================================================
# CLEANUP CACHE
# ======================================================
def cleanup_cache():
    now = time.time()

    if not os.path.exists(CACHE_DIR):
        return

    # ==========================================
    # DELETE OLD FILES
    # ==========================================
    for root, _, files in os.walk(
        CACHE_DIR
    ):

        for file in files:
            path = os.path.join(
                root,
                file
            )

            try:
                age = (
                    now
                    - os.path.getmtime(path)
                )

                if age > TTL:
                    os.remove(path)
                    print(
                        f"Deleted cache: {path}"
                    )

            except Exception as e:
                print(
                    "Cleanup failed:",
                    e
                )

    # ==========================================
    # REMOVE EMPTY DIRS
    # ==========================================
    for root, dirs, files in os.walk(
        CACHE_DIR,
        topdown=False
    ):
        if not dirs and not files:
            try:
                os.rmdir(root)
            except:
                pass

# ======================================================
# START SCHEDULER
# ======================================================
scheduler = BackgroundScheduler()

scheduler.add_job(
    cleanup_cache,
    "interval",
    minutes=5
)

scheduler.start()