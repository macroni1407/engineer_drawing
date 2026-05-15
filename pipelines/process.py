import os
import cv2
import time
import json
import hashlib

from redis_client import use_redis

from ocr import (
    get_ocr,
    html_to_excel_zip,
    texts_to_txt_buffer,
    save_crops_to_zip,
)

from models import ensemble_inference
from visualization import visualize_predictions
from inferences import crop_objects_from_masks

from cache_manager import *
from configs import *

# =========================================================
# REDIS
# =========================================================

redis = use_redis()

REDIS_TTL = 300

# =========================================================
# MAIN PROCESS
# =========================================================

def process_image(input_img):
    now = time.time()

    if input_img is None:
        return None, None, None, None

    # =====================================================
    # IMAGE NAME
    # =====================================================
    image_name = os.path.basename(
        input_img
    )

    # =====================================================
    # READ IMAGE
    # =====================================================
    img_bgr = cv2.imread(input_img)

    if img_bgr is None:
        return None, None, None, None

    img_rgb = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2RGB,
    )

    # =====================================================
    # HASH
    # =====================================================
    image_hash = hashlib.md5(
        img_bgr.tobytes()
    ).hexdigest()

    # =====================================================
    # CACHE LOAD
    # =====================================================
    cached = redis.get(image_hash)

    if cached:
        try:
            cached_data = json.loads(
                cached
            )

            result = load_result_cache(
                cached_data.get(
                    "result_path"
                )
            )

            if result is not None:
                print("CACHE HIT")
                gallery_images = (
                    load_gallery_cache(
                        cached_data.get(
                            "gallery_images",
                            []
                        )
                    )
                )

                return (
                    result,
                    gallery_images,
                    cached_data.get(
                        "metadata_json"
                    ),
                    cached_data.get(
                        "zip_path"
                    ),
                )

        except Exception as e:
            print(
                "Cache load failed:",
                e
            )

    # =====================================================
    # INFERENCE
    # =====================================================
    instances = ensemble_inference(
        image=img_bgr,
        scales=TTA_SCALES,
        flip=TTA_FLIP,
        score_thresh=SCORE_THRESH,
        box_nms_thresh=BOX_NMS_THRESH,
        mask_nms_thresh=MASK_NMS_THRESH,
        max_detections=MAX_DETECTIONS,
    )

    # =====================================================
    # VISUALIZATION
    # =====================================================
    result = visualize_predictions(
        img_rgb,
        instances,
    )

    # =====================================================
    # CROP
    # =====================================================
    crops, metadata_json = (
        crop_objects_from_masks(
            img_rgb,
            instances,
            image_name=image_name,
        )
    )

    # =====================================================
    # OCR
    # =====================================================
    ocr_model = get_ocr()

    object_map = {
        obj["id"]: obj
        for obj in metadata_json["objects"]
    }

    files = []
    for crop, label_name, _, obj_id in crops:
        obj_val = object_map.get(obj_id)

        if obj_val is None:
            continue

        print(label_name)
        if label_name not in [
            "table",
            "note"
        ]:
            continue

        try:
            info = ocr_model.predict(
                input=crop
            )

        except Exception as e:
            print(
                "OCR failed:",
                e
            )
            continue

        parsing_res = info[0][
            "parsing_res_list"
        ]

        # =================================================
        # TABLE
        # =================================================
        if label_name == "table":
            table_contents = []
            for value in parsing_res:
                print("label: ", value.label)
                print("content: ", value.content)
                print("type: ", type(value.content))
                if (
                    "<table"
                    not in value.content.lower()
                ):
                    continue

                table_contents.append(
                    value.content
                )

                excel_buffer = (
                    html_to_excel_zip(
                        value.content
                    )
                )

                files.append({
                    "buffer": excel_buffer,
                    "type": "xlsx",
                })

            obj_val["content"] = (
                table_contents
            )

        # =================================================
        # NOTE
        # =================================================
        elif label_name == "note":
            notes = []
            for value in parsing_res:
                print("label: ", value.label)
                print("content: ", value.content)
                print("type: ", type(value.content))
                notes.append(
                    value.content
                )

            obj_val["content"] = notes

            if len(notes) > 0:
                txt_buffer = (
                    texts_to_txt_buffer(
                        notes
                    )
                )

                files.append({
                    "buffer": txt_buffer,
                    "type": "txt",
                })

    # =====================================================
    # GALLERY RUNTIME
    # =====================================================
    gallery_images = [
        (
            img,
            f"{label_name}: {score}"
        )

        for img,
        label_name,
        score,
        _ in crops
    ]

    # =====================================================
    # SAVE GALLERY CACHE
    # =====================================================
    gallery_cache = save_gallery_cache(
        image_hash,
        crops,
    )

    # =====================================================
    # ZIP
    # =====================================================
    zip_path = save_crops_to_zip(
        crops,
        files,
        metadata_json,
        image_hash,
    )

    # =====================================================
    # SAVE RESULT CACHE
    # =====================================================
    result_path = save_result_cache(
        image_hash,
        result
    )

    # =====================================================
    # REDIS SAVE
    # =====================================================
    cache_data = {
        "result_path": result_path,
        "metadata_json": metadata_json,
        "zip_path": zip_path,
        "gallery_images": gallery_cache,
    }

    try:
        redis.set(
            image_hash,
            json.dumps(cache_data),
            ex=REDIS_TTL
        )

    except Exception as e:
        print(
            "Redis save failed:",
            e
        )

    print(
        f"time: {time.time() - now:.2f}s"
    )

    return (
        result,
        gallery_images,
        metadata_json,
        zip_path,
    )