from pathlib import Path
import time
import traceback
import torch
from openslide import open_slide
from PIL import Image
import os
import numpy as np
import cv2

from utils import cucim_image_info, slide_info, slide_process_single, mask_to_geojson, get_td_preprocessing, get_ad_preprocessing, make_class_map
from utils import colors_QC7 as colors

from onnx_export import load_sd_onnx_session, load_td_onnx_session

Image.MAX_IMAGE_PIXELS = 1000000000


# ======== [STAGE 1] ========
import segmentation_models_pytorch as smp
from openslide import OpenSlide


def load_td_model(ENCODER_MODEL_TD, ENCODER_MODEL_TD_WEIGHTS, MODEL_TD_PATH, DEVICE="cuda"):
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER_MODEL_TD, ENCODER_MODEL_TD_WEIGHTS)

    model = smp.UnetPlusPlus(
        encoder_name=ENCODER_MODEL_TD,
        encoder_weights=ENCODER_MODEL_TD_WEIGHTS,
        classes=2,
        activation=None,
    )

    model.load_state_dict(torch.load(MODEL_TD_PATH, map_location='cpu'))
    model.to(DEVICE)
    model.eval()
    return preprocessing_fn, model


def tissue_detection(slide_path, ENCODER_MODEL_TD, ENCODER_MODEL_TD_WEIGHTS, MODEL_TD_PATH, MPP_MODEL_TD, M_P_S_MODEL_TD, DEVICE):
    start_time = time.perf_counter()
    slide_name = Path(slide_path).name

    onnx_path = "/workspace/application/tissue_detection.onnx"
    preprocessing_fn, session = load_td_onnx_session(ENCODER_MODEL_TD, ENCODER_MODEL_TD_WEIGHTS, onnx_path)
    # preprocessing_fn, model = load_td_model(ENCODER_MODEL_TD, ENCODER_MODEL_TD_WEIGHTS, MODEL_TD_PATH, DEVICE)

    slide = OpenSlide(slide_path)

    w_l0, h_l0 = slide.level_dimensions[0]
    mpp = round(float(slide.properties["openslide.mpp-x"]), 4)
    reduction_factor = MPP_MODEL_TD / mpp
    print(w_l0, h_l0, reduction_factor, w_l0 // reduction_factor, h_l0 // reduction_factor)

    tmp_dir = ""
    image_or = slide.get_thumbnail((w_l0 // reduction_factor, h_l0 // reduction_factor))
    image_or.save(tmp_dir + slide_name + ".jpg", quality = 80)

    '''
    As tissue detector was trained on jpeg compressed images - we have to reproduce this step.
    Otherwise it functions suboptimal.
    '''

    image = np.array(image_or)
    print("TD model Input Image Size: ", image.shape)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
    _, image = cv2.imencode('.jpg', image, encode_param)
    image = cv2.imdecode(image, 1)
    image = Image.fromarray(image)

    width, height = image.size

    wi_n = width // M_P_S_MODEL_TD
    he_n = height // M_P_S_MODEL_TD

    overhang_wi = width - wi_n * M_P_S_MODEL_TD
    overhang_he = height - he_n * M_P_S_MODEL_TD

    print('Overhang (< 1 patch) for width and height: ', overhang_wi, ',', overhang_he)

    p_s = M_P_S_MODEL_TD

    for h in range(he_n + 1):
        for w in range(wi_n + 1):
            if w != wi_n and h != he_n:
                image_work = image.crop((w * p_s, h * p_s, (w + 1) * p_s, (h + 1) * p_s))
            elif w == wi_n and h != he_n:
                image_work = image.crop((width - p_s, h * p_s, width, (h + 1) * p_s))
            elif w != wi_n and h == he_n:
                image_work = image.crop((w * p_s, height - p_s, (w + 1) * p_s, height))
            else:
                image_work = image.crop((width - p_s, height - p_s, width, height))

            image_pre = get_td_preprocessing(image_work, preprocessing_fn)
            # x_tensor = torch.from_numpy(image_pre).to(DEVICE).unsqueeze(0)
            # print(x_tensor.shape)
            # predictions = model.predict(x_tensor)
            # predictions = (predictions.squeeze().cpu().numpy())

            x = image_pre.astype(np.float32)
            if x.ndim == 3:
                # image_pre가 (C, H, W) 또는 (H, W, C)냐에 따라 정리 필요
                # 여기서는 기존 코드에서 torch.from_numpy(image_pre) 바로 모델에 넣었다고 했으니
                # 그 shape을 그대로 따른다고 가정 (예: (C, H, W)).
                x = np.expand_dims(x, axis=0)  # (1, C, H, W)

            # 2) onnxruntime 입력 이름
            input_name = session.get_inputs()[0].name

            # 3) 추론
            ort_outs = session.run(None, {input_name: x})
            output = ort_outs[0]  # (1, classes, H, W)

            # 4) squeeze해서 기존과 동일한 형태로 맞추기
            predictions = np.squeeze(output)  # (classes, H, W) 또는 (H, W) depending on how much squeeze

            mask = np.argmax(predictions, axis=0).astype('int8')

            class_mask = make_class_map(mask, colors)

            if w == 0:
                temp_image = mask
                temp_image_class_map = class_mask
            elif w == wi_n:
                mask = mask[:, p_s - overhang_wi:p_s]
                temp_image = np.concatenate((temp_image, mask), axis=1)
                class_mask = class_mask[:, p_s - overhang_wi:p_s, :]
                temp_image_class_map = np.concatenate((temp_image_class_map, class_mask), axis=1)
            else:
                temp_image = np.concatenate((temp_image, mask), axis=1)
                temp_image_class_map = np.concatenate((temp_image_class_map, class_mask), axis=1)
        if h == 0:
            end_image = temp_image
            end_image_class_map = temp_image_class_map
        elif h == he_n:
            temp_image = temp_image [p_s - overhang_he:p_s,]
            end_image = np.concatenate((end_image, temp_image), axis=0)
            temp_image_class_map = temp_image_class_map [p_s - overhang_he:p_s, :, :]
            end_image_class_map = np.concatenate((end_image_class_map, temp_image_class_map), axis=0)
        else:
            end_image = np.concatenate((end_image, temp_image), axis=0)
            end_image_class_map = np.concatenate((end_image_class_map, temp_image_class_map), axis=0)

    mask_img_pil = Image.fromarray(end_image)
    print("Mask Image Result: ", end_image.shape)
    print("Tissue Detection Duration: ", time.perf_counter() - start_time)

    return mask_img_pil



# ======== [STAGE 2] ========

def load_ad_model(MODEL_QC_PATH, map_location="cuda", weights_only=False):
    model_prim = torch.load(MODEL_QC_PATH, map_location=map_location, weights_only=False)

    return model_prim


def artifact_detection(slide_path, MODEL_QC_PATH, tis_det_map, ENCODER_MODEL, ENCODER_MODEL_WEIGHTS, MPP_MODEL=1.0, M_P_S_MODEL=512, BACK_CLASS=7):
    model = load_ad_model(MODEL_QC_PATH, map_location="cuda", weights_only=False)
    MODEL_AD_ONNX_PATH = "/workspace/application/artifact_detection.onnx"
    session = load_sd_onnx_session(MODEL_AD_ONNX_PATH, use_gpu=True)

    slide_name = Path(slide_path).name
    start_time = time.perf_counter()

    print("")
    print("Processing:", slide_name)

    # Open slide
    cuimg = CuImage(slide_path)

    # GET SLIDE INFO
    # p_s, patch_n_w_l0, patch_n_h_l0, mpp, w_l0, h_l0 = slide_info(slide, M_P_S_MODEL, MPP_MODEL)
    p_s, patch_n_w_l0, patch_n_h_l0, mpp, w_l0, h_l0 = cucim_image_info(cuimg, M_P_S_MODEL, MPP_MODEL)

    # LOAD TISSUE DETECTION MAP
    '''
    Tissue detection map is generated on MPP = 10
    This map is used for on-fly control of the necessity of model inference.
    Two variants: reduced version with perfect correlation or full version scaled to working MPP of the tumor detection model
    Classes: 0 - tissue, 1 - background
    '''

    tis_det_np = np.array(tis_det_map, dtype=np.uint16)
    tis_det_np = (tis_det_np / 256).astype(np.uint8)  # 16비트를 8비트로 스케일링
    tis_det_map = Image.fromarray(tis_det_np, mode='L')  # 8비트 그레이스케일 변환

    tis_det_map_mpp = np.array(tis_det_map.resize((int(w_l0 * mpp / MPP_MODEL), int(h_l0 * mpp / MPP_MODEL)), Image.Resampling.LANCZOS))
    _, full_mask_np = slide_process_single(model, tis_det_map_mpp, cuimg, patch_n_w_l0, patch_n_h_l0, p_s,
                                            M_P_S_MODEL, colors, ENCODER_MODEL,
                                            ENCODER_MODEL_WEIGHTS, "cuda", BACK_CLASS, MPP_MODEL, mpp, w_l0, h_l0)

    # Timer stop
    print("Artifact Detection Duration: ", time.perf_counter() - start_time)

    geojson_path = "/workspace/01_WSI_inference_OPENSLIDE_QC/output/result.geojson"
    geojson = mask_to_geojson(full_mask_np, output_path=geojson_path)

    return geojson, full_mask_np.shape



def pipeline(slide_path):
    # DEVICE
    DEVICE = 'cuda'
    '''
    'cuda' - NVIDIA GPU card
    'mps'    - APPLE Silicon
    '''

    # MODEL TISSUE DETECTION:
    MODEL_TD_PATH = '/workspace/01_WSI_inference_OPENSLIDE_QC/models/td/Tissue_Detection_MPP10.pth'
    MPP_MODEL_TD = 10
    M_P_S_MODEL_TD = 512
    ENCODER_MODEL_TD = 'timm-efficientnet-b0'
    ENCODER_MODEL_TD_WEIGHTS = 'imagenet'

    # MODEL ARTIFACT DETECTION:
    MODEL_QC_PATH = '/workspace/01_WSI_inference_OPENSLIDE_QC/models/qc/GrandQC_MPP1.pth'
    ENCODER_MODEL = 'timm-efficientnet-b0'
    ENCODER_MODEL_WEIGHTS = 'imagenet'

    # OVERLAY PARAMETERS (TRANSPARENCY)
    OVER_IMAGE = 0.7    # % original image
    OVER_MASK = 0.3     # % segmentation mask

    # COLORS for MASK
    colors = [[50, 50, 250],    # BLUE: TISSUE
            [128, 128, 128]]  # GRAY: BACKGROUND

    tissue_mask = tissue_detection(slide_path, ENCODER_MODEL_TD, ENCODER_MODEL_TD_WEIGHTS, MODEL_TD_PATH, MPP_MODEL_TD, M_P_S_MODEL_TD, DEVICE)
    print("mask img size: ", tissue_mask.size)
    geojson, (mask_h, mask_w) = artifact_detection(slide_path, MODEL_QC_PATH, tissue_mask, ENCODER_MODEL, ENCODER_MODEL_WEIGHTS, MPP_MODEL=1.0, M_P_S_MODEL=512, BACK_CLASS=7)

    return geojson, mask_w, mask_h


CLASS_COLOR_MAP = {
    2: [255, 99, 71],     #ART_FOLD: orange
    3: [0, 255, 0],   #ART_DARKSPOT: green
    4: [255, 0, 0],   #ART_PEN: red
    5: [255, 0, 255],   #ART_EDGE: pink
    6: [75, 0, 130],      # ART_FOCUS: violet
}

def get_color_for_class(class_id: int):
    return CLASS_COLOR_MAP.get(class_id, (128, 128, 128))  # default: gray


import cv2
import numpy as np

def overlay_class_resized_img(origin_np, geojson, output_path, ds_factor=8):
    """
    origin_np : 원본 이미지 (H, W) 또는 (H, W, C)
    geojson   : Polygon 좌표는 (x, y) - 보통 mask 또는 원본 기준
    factor    : geojson 좌표 → 원본 이미지 좌표로 가는 스케일 (기존에 쓰던 값)
    ds_factor : 최종 출력 이미지에서 원본 대비 축소 배율 (기본 8)
    """

    # 원본 크기
    h, w = origin_np.shape[:2]

    # 2) 좌표 스케일: geojson 좌표 → resized_img 좌표
    #    (기존 factor: geojson → 원본, downsample: 원본 → resized)
    #    => geojson → resized = factor / downsample
    coord_scale = 1

    for feature in geojson["features"]:
        geom = feature.get("geometry", {})
        if geom.get("type") != "Polygon":
            continue

        props = feature.get("properties", {})
        class_id = props.get("class_id", None)
        color = get_color_for_class(class_id)  # BGR 튜플이라고 가정

        for ring in geom["coordinates"]:
            pts_list = []

            for x, y in ring:
                # geojson 좌표 → resized 이미지 좌표
                x_resized = int(x * coord_scale)
                y_resized = int(y * coord_scale)

                # 범위 체크 (resized 이미지 기준)
                if 0 <= x_resized < w and 0 <= y_resized < h:
                    pts_list.append([x_resized, y_resized])

            if len(pts_list) < 2:
                continue  # 점이 너무 적으면 스킵

            pts = np.array(pts_list, dtype=np.int32).reshape((-1, 1, 2))

            # 테두리만 그리고 싶으면:
            cv2.polylines(region_np, [pts], isClosed=True, color=color, thickness=15)

    resized = cv2.resize(region_np, (region_np.shape[1] // ds_factor, region_np.shape[0] // ds_factor))
    print("최종 저장되는 이미지 크기", resized.shape)
    cv2.imwrite(output_path, resized)
    return output_path


from cucim import CuImage

if __name__ == "__main__":
    start = time.perf_counter()
    slide_path = '/workspace/slides/TCGA-BH-A0DQ-01Z,1A.3A,.svs'
    # slide_path = '/workspace/slides/PIT-01-BRBX-00891-IMH-001.svs'
    # slide_path = '/workspace/slides/PIT-01-ESBX-01163-IMH-001.svs'

    cuimg = CuImage(slide_path)
    p_s, patch_n_w_l0, patch_n_h_l0, mpp, w_l0, h_l0 = cucim_image_info(cuimg, m_p_s=512, mpp_model=1.0)
    w, h = cuimg.size()[:2]
    resolutions = cuimg.resolutions
    level_dimensions = resolutions["level_dimensions"]
    level_count = resolutions["level_count"]

    print("level dimensions", level_dimensions)
    downsample = level_dimensions[level_count - 1]

    ds_h_factor = h / downsample[0]
    ds_w_factor = w / downsample[1]

    print("downsample factor", ds_h_factor, ds_w_factor)

    ds_region = cuimg.read_region(
        location=[0, 0], size=level_dimensions[level_count - 1], level=level_count - 1
    )
    region_np = np.array(ds_region)
    print("CuImage 불러오기", time.perf_counter() - start)

    start = time.perf_counter()
    try:
        geojson, mask_w, mask_h = pipeline(slide_path)
    except:
        traceback.print_exc()

    print("QC 처리 완료", time.perf_counter() - start)

    # print(geojson)
    print("♠️♦️♠️♦️")

    start = time.perf_counter()

    output_path = "/workspace/01_WSI_inference_OPENSLIDE_QC/output/result_pit_3.png"
    region_np = cv2.resize(region_np, (mask_w, mask_h))
    print("overlay 할 원본 이미지", region_np.shape)
    output_path = overlay_class_resized_img(region_np, geojson, output_path, ds_factor=int(ds_w_factor // 4))
    print("overlay 완료", time.perf_counter() - start)
