from pathlib import Path
import time
import traceback
from wsi_colors import colors_QC7 as colors
import torch
import argparse
from openslide import open_slide
from PIL import Image
import os
from wsi_slide_info import slide_info
from wsi_process import slide_process_single, mask_to_geojson
from wsi_maps import make_overlay
import numpy as np
import timeit
import cv2
Image.MAX_IMAGE_PIXELS = 1000000000


# MODEL(S)
MODEL_QC_DIR = '/workspace/01_WSI_inference_OPENSLIDE_QC/models/qc/'
MPP_MODEL = 1.0
MODEL_QC_NAME = 'GrandQC_MPP1.pth'
M_P_S_MODEL = 512
ENCODER_MODEL = 'timm-efficientnet-b0'
ENCODER_MODEL_WEIGHTS = 'imagenet'

# CLASSES
BACK_CLASS = 7

OUTPUT_DIR = ""
geojson_root = os.path.join(OUTPUT_DIR, "geojson_qc")
os.makedirs(geojson_root, exist_ok=True)


def load_model(MODEL_QC_DIR, MODEL_QC_NAME, map_location="cuda", weights_only=False):
    model_prim = torch.load(MODEL_QC_DIR + MODEL_QC_NAME, map_location=map_location, weights_only=False)

    return model_prim


def processing_qc(slide_path):
    model = load_model(MODEL_QC_DIR, MODEL_QC_NAME, map_location="cuda", weights_only=False)

    slide_name = Path(slide_path).name
    start_time = time.perf_counter()

    print("")
    print("Processing:", slide_name)

    # Open slide
    slide = open_slide(slide_path)

    # GET SLIDE INFO
    p_s, patch_n_w_l0, patch_n_h_l0, mpp, w_l0, h_l0, obj_power = slide_info(slide, M_P_S_MODEL, MPP_MODEL)

    # LOAD TISSUE DETECTION MAP
    tis_det_map = Image.open(os.path.join(OUTPUT_DIR, 'tis_det_mask', slide_name + '_MASK.png'))

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
    map, full_mask = slide_process_single(model, tis_det_map_mpp, slide, patch_n_w_l0, patch_n_h_l0, p_s,
                                            M_P_S_MODEL, colors, ENCODER_MODEL,
                                            ENCODER_MODEL_WEIGHTS, "cuda", BACK_CLASS, MPP_MODEL, mpp, w_l0, h_l0)

    # Timer stop
    stop = timeit.default_timer()
    print("Duration: ", time.perf_counter() - start_time)
