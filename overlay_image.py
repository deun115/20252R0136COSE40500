import traceback

from PIL import Image
import numpy as np
import matplotlib.cm as cm


def overlay_image(base_np, overlay_np) -> np.ndarray:
    try:
        base = Image.fromarray(base_np).convert("RGB")
        overlay = Image.fromarray(overlay_np)

        # overlay 이미지가 grayscale인 경우 viridis cmap 적용
        if overlay.mode == "L":
            # [0, 1]로 정규화
            norm_overlay = (overlay_np - overlay_np.min()) / (overlay_np.max() - overlay_np.min())
            viridis_colored = cm.viridis(norm_overlay)

            # RGB 채널만 사용
            viridis_rgb = (viridis_colored[:, :, :3] * 255).astype(np.uint8)

            # RGB 모드로 변환
            overlay = Image.fromarray(viridis_rgb).convert("RGB")

        else:
            overlay = overlay.convert("RGB")

        overlay = overlay.resize(base.size, Image.Resampling.BILINEAR)

        # 이미지 블렌딩 (alpha=0.5)
        overlay_with_alpha = Image.blend(base, overlay, alpha=0.5)
        result_np = np.array(overlay_with_alpha)

        return result_np
    except Exception as e:
        traceback.print_exc()


def main(vmeta) -> np.ndarray:
    base_np, overlay_np = None, None
    for f_key in vmeta:
        for s_key in vmeta[f_key]:
            if f_key == "origin" and "img-" in s_key:
                base_np = vmeta[f_key][s_key]
            elif "pred-" in f_key and "img-" in s_key:
                overlay_np = vmeta[f_key][s_key]

    result = overlay_image(base_np, overlay_np)
    return result


input_path = "/workspace/vience.jpeg"


# overlay 이미지 만들기

from PIL import Image
import numpy as np
import cv2

input_img = Image.open(input_path)
input_np = np.array(input_img)

grayscale_img = input_img.convert('L')   # 8비트 흑백
grayscale_np = np.array(grayscale_img)
# output_np = np.max(grayscale_np) - grayscale_np

result_np = overlay_image(input_np, grayscale_np)

cv2.imwrite("/workspace/vience-overlay.jpg", result_np, [cv2.IMWRITE_JPEG_QUALITY, 90])