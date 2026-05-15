import os
import io
import json
import zipfile

from PIL import Image
from cache_manager import ZIP_DIR

# ======================================================
# SAVE ZIP
# ======================================================

def save_crops_to_zip(
    crops,
    files,
    metadata_json,
    image_hash,
):
    os.makedirs(
        ZIP_DIR,
        exist_ok=True
    )

    zip_path = os.path.join(
        ZIP_DIR,
        f"{image_hash}.zip"
    )

    try:
        # already cached
        if os.path.exists(zip_path):
            return zip_path

        with zipfile.ZipFile(
            zip_path,
            "w",
            zipfile.ZIP_DEFLATED,
        ) as zipf:
            root_folder = "cropped_objects"

            # ==========================================
            # metadata.json
            # ==========================================
            metadata_bytes = json.dumps(
                metadata_json,
                indent=2,
                ensure_ascii=False,
            ).encode("utf-8")

            zipf.writestr(
                f"{root_folder}/metadata.json",
                metadata_bytes,
            )

            # ==========================================
            # crop images
            # ==========================================
            for idx, (
                img_rgb,
                label,
                _,
                _
            ) in enumerate(crops):
                img_buffer = io.BytesIO()

                Image.fromarray(
                    img_rgb
                ).save(
                    img_buffer,
                    format="PNG",
                )

                img_buffer.seek(0)

                zipf.writestr(
                    f"{root_folder}/images/{label}_{idx}.png",
                    img_buffer.getvalue(),
                )

            # ==========================================
            # txt + excel
            # ==========================================
            note_count = 0
            table_count = 0

            for file_info in files:
                file_buffer = file_info["buffer"]
                file_type = file_info["type"]

                file_buffer.seek(0)

                if file_type == "xlsx":
                    filename = (
                        f"{root_folder}/excels/"
                        f"table_{table_count}.xlsx"
                    )

                    table_count += 1

                else:
                    filename = (
                        f"{root_folder}/notes/"
                        f"note_{note_count}.txt"
                    )

                    note_count += 1

                zipf.writestr(
                    filename,
                    file_buffer.getvalue(),
                )

        return zip_path

    except Exception as e:
        print(
            "Save zip failed:",
            e
        )
        return None