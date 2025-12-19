import json
from pathlib import Path
import cv2
import numpy as np

# class_id 별 색상 (BGR)
CLASS_COLOR_MAP = {
    1: (0, 0, 255),      # red   (BGR)
    2: (0, 255, 0),      # green
    3: (255, 0, 0),      # blue
    4: (0, 255, 255),    # yellow
    5: (255, 0, 255),    # magenta
    6: (255, 255, 0),    # cyan
    7: (0, 165, 255),    # orange
}

def get_color_for_class(class_id: int):
    return CLASS_COLOR_MAP.get(class_id, (128, 128, 128))  # default: gray

geojson_path = Path("/workspace/output/geojson_qc/TCGA-BH-A0DQ-01Z,1A.3A,.svs.geojson")

# 1) 이미지 로드 (BGR)
slide_path = "/workspace/slides/TCGA-BH-A0DQ-01Z,1A.3A,.svs"
from openslide import OpenSlide

slide = OpenSlide(slide_path)
image = slide.read_region((0, 0), 2, (slide.level_dimensions[2][0], slide.level_dimensions[2][1]))

print("image size:", image.size)
h, w = image.size
image_np = np.array(image)

# 2) GeoJSON 로드
with open(geojson_path, "r", encoding="utf-8") as f:
    geojson = json.load(f)

# 3) feature별 polygon 오버레이
for feature in geojson["features"]:
    geom = feature.get("geometry", {})
    if geom.get("type") != "Polygon":
        continue

    props = feature.get("properties", {})
    class_id = props.get("class_id", None)
    color = get_color_for_class(class_id)

    for ring in geom["coordinates"]:
        # ring: [[x, y], ...] → int32 numpy array로 변환
        pts = np.array([[int(x // 16), int(y // 16)] for x, y in ring if x // 16 <= w and y // 16 <= h], dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))

        # 꽉 채우는 폴리곤
        # cv2.fillPoly(image, [pts], color)

        # 테두리만 그리고 싶으면:
        cv2.polylines(image_np, [pts], isClosed=True, color=color, thickness=15)

# 4) 저장
out_path = Path("/workspace/visualization/overlay_cv2_tcga_level2.jpg")
cv2.imwrite(str(out_path), image_np, [cv2.IMWRITE_JPEG_QUALITY, 90])
print("saved to", out_path)


from pathlib import Path
from PIL import Image
from openslide import open_slide

Image.MAX_IMAGE_PIXELS = None


from pathlib import Path
import pyvips

# if __name__ == "__main__":
#     in_path = Path("/workspace/visualization/overlay_cv2_tcga.png")
#     out_path = Path("/workspace/visualization/overlay_cv2_tcga_downsample.png")

#     # 원본 이미지 로드 (스트리밍 방식)
#     image = pyvips.Image.new_from_file(str(in_path), access="sequential")

#     print("original size:", image.width, image.height)

#     # 16배 축소 (width, height 각각 1/16)
#     factor = 16
#     down = image.shrink(factor, factor)

#     print("downsampled size:", down.width, down.height)

#     # 저장
#     down.write_to_file(str(out_path))
