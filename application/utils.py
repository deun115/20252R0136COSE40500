import time
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import torch
from tqdm import tqdm
import cv2
import json
from cucim import CuImage


# EXTRACTION OF META-DATA FROM SLIDE
colors_QC7 = [
    [128, 128, 128],     #ART_NORM: white
    [255, 99, 71],     #ART_FOLD: orange
    [0, 255, 0],   #ART_DARKSPOT: green
    [255, 0, 0],   #ART_PEN: red
    [255, 0, 255],   #ART_EDGE: pink
    [75, 0, 130],      # ART_FOCUS: violet
    [255, 255, 255]   #BACKGROUND
]


def slide_info(slide, m_p_s, mpp_model):
    # Microne per pixel
    mpp = float(slide.properties["openslide.mpp-x"])
    p_s = int(mpp_model / mpp * m_p_s)

    # Extract and save dimensions of level [0]
    dim_l0 = slide.level_dimensions[0]
    w_l0 = dim_l0[0]
    h_l0 = dim_l0[1]

    # Calculate number of patches to process
    patch_n_w_l0 = int(w_l0 / p_s)
    patch_n_h_l0 = int(h_l0 / p_s)

    # Number of levels
    num_level = slide.level_count

    # Level downsamples
    down_levels = slide.level_downsamples

    # Output BASIC DATA
    print("")
    print("Basic data about processed whole-slide image")
    print("")
    print("Number of levels: ", num_level)
    print("Level downsamples: ", down_levels)
    print("Microns per pixel (slide):", mpp)
    print("Height: ", h_l0)
    print("Width: ", w_l0)
    print("Model patch size at slide MPP: ", p_s, "x", p_s)
    print("Width - number of patches: ", patch_n_w_l0)
    print("Height - number of patches: ", patch_n_h_l0)
    print("Overall number of patches / slide (without tissue detection): ", patch_n_w_l0 * patch_n_h_l0)

    return p_s, patch_n_w_l0, patch_n_h_l0, mpp, w_l0, h_l0


def cucim_image_info(cuimg: CuImage, m_p_s, mpp_model):
    # Microne per pixel
    mpp = float(cuimg.metadata['aperio']['MPP'])
    p_s = int(mpp_model / mpp * m_p_s)

    # Extract and save dimensions of level [0]
    resolutions = cuimg.resolutions
    dim_l0 = resolutions["level_dimensions"][0]
    w_l0 = dim_l0[0]
    h_l0 = dim_l0[1]

    # Calculate number of patches to process
    patch_n_w_l0 = int(w_l0 / p_s)
    patch_n_h_l0 = int(h_l0 / p_s)

    # Number of levels
    num_level = resolutions["level_count"]

    # Level downsamples
    down_levels = resolutions["level_downsamples"]

    # Output BASIC DATA
    print("")
    print("Basic data about processed whole-slide image")
    print("")
    print("Number of levels: ", num_level)
    print("Level downsamples: ", down_levels)
    print("Microns per pixel (slide):", mpp)
    print("Height: ", h_l0)
    print("Width: ", w_l0)
    print("Model patch size at slide MPP: ", p_s, "x", p_s)
    print("Width - number of patches: ", patch_n_w_l0)
    print("Height - number of patches: ", patch_n_h_l0)
    print("Overall number of patches / slide (without tissue detection): ", patch_n_w_l0 * patch_n_h_l0)

    return p_s, patch_n_w_l0, patch_n_h_l0, mpp, w_l0, h_l0



# MAIN LOOP TO PROCESS WSI
#Helper functions
def to_tensor_x(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_ad_preprocessing(image, preprocessing_fn, model_size):
    if image.size != model_size:
        image = image.resize(model_size)
        print('resized')
    image = np.array(image)
    x = preprocessing_fn(image)
    x = to_tensor_x(x)
    return x


def make_1class_map_thr(mask, class_colors):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(1, len(class_colors)+1):
        idx = mask == l
        r[idx] = class_colors [l-1][0]
        g[idx] = class_colors [l-1][1]
        b[idx] = class_colors [l-1][2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb


def slide_process_single(model, tis_det_map_mpp, cuimg, patch_n_w_l0, patch_n_h_l0, p_s, m_p_s, colors,
                         ENCODER_MODEL_1,ENCODER_WEIGHTS, DEVICE, BACK_CLASS, MPP_MODEL_1, mpp, w_l0, h_l0):
    '''
    Tissue detection map is generated under MPP = 4, therefore model patch size of (512,512) corresponds to tis_det_map patch
    size of (128,128).
    '''

    model_size = (m_p_s, m_p_s)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER_MODEL_1, ENCODER_WEIGHTS)

    # Start loop
    end_rows = []

    for he in tqdm(range(patch_n_h_l0), total=patch_n_h_l0):
        # ------------------------------
        # 1) he(세로 인덱스) 기반으로 한 줄 strip 읽기
        # ------------------------------
        h = he * p_s + 1
        if he == 0:
            h = 0

        # 기존 코드에서 w는 0, p_s+1, 2*p_s+1, ... 이런 식이었으므로
        # 이를 모두 커버하려면 +1 여유를 둔 폭으로 읽어두는 것이 안전함
        row_width = patch_n_w_l0 * p_s + 1

        # 한 번만 read_region 해서 strip 전체를 읽어둠
        row_region = cuimg.read_region(
            location=[0, h],              # level-0 기준 (x=0, y=h)에서 시작
            size=(row_width, p_s),        # width, height
            level=0
        )
        row_region_np = np.asarray(row_region)  # (p_s, row_width, C)

        # tissue detection map에서 이 he에 해당하는 세로 구간을 미리 잘라둠
        td_row = tis_det_map_mpp[he * m_p_s:(he + 1) * m_p_s, :]

        row_masks = []

        # ------------------------------
        # 2) wi 루프: strip 안에서 numpy 슬라이싱
        # ------------------------------
        for wi in range(patch_n_w_l0):
            # 기존 코드의 w 계산을 그대로 유지
            w = wi * p_s + 1
            if wi == 0:
                w = 0

            # 2-1) tissue detection patch (기존과 동일)
            td_patch = td_row[:, wi * m_p_s:(wi + 1) * m_p_s]

            if td_patch.shape != (512, 512):
                # td_patch padding (incase td_patch does not equal (512,512))
                original_shape = td_patch.shape
                desired_shape = (512, 512)

                # Calculate padding needed
                padding = [(0, desired_shape[i] - original_shape[i]) for i in range(2)]
                td_patch_ = np.pad(td_patch, padding, mode='constant')
            else:
                td_patch_ = td_patch

            # 2-2) td_patch 기반으로 segmentation 필요 여부 판단
            if np.count_nonzero(td_patch == 0) > 50:  # here change to check of segmentation map
                # 2-3) strip에서 이미지 패치 슬라이싱
                # row_region_np shape: (p_s, row_width, C)
                # w부터 w + p_s까지 잘라서 해당 wi 패치 이미지 획득
                patch_np = row_region_np[:, w:w + p_s, :]  # (p_s, p_s, C) 가 되도록 설계

                # PIL 이미지로 변환
                work_patch = Image.fromarray(patch_np).convert('RGB')

                # Resize to model patch size (기존과 동일)
                work_patch = work_patch.resize((m_p_s, m_p_s), Image.Resampling.LANCZOS)

                # 전처리
                image_pre = get_ad_preprocessing(work_patch, preprocessing_fn, model_size)
                x_tensor = torch.from_numpy(image_pre).to(DEVICE).unsqueeze(0)
                predictions = model.predict(x_tensor)
                predictions = (predictions.squeeze().cpu().numpy())

                mask_raw = np.argmax(predictions, axis=0).astype('int8')
                mask = np.where(td_patch_ == 1, BACK_CLASS, mask_raw)
            else:
                # tissue가 거의 없으면 통으로 BACK_CLASS
                mask = np.full((512, 512), BACK_CLASS, dtype='int8')

            row_masks.append(mask)

        # ------------------------------
        # 3) wi 방향으로 하나의 row stitch
        # ------------------------------
        temp_image = np.concatenate(row_masks, axis=1)  # (512, 512 * patch_n_w_l0)
        end_rows.append(temp_image)

    # ------------------------------
    # 4) he 방향으로 전체 stitch
    # ------------------------------
    end_image = np.concatenate(end_rows, axis=0)  # (512 * patch_n_h_l0, 512 * patch_n_w_l0)

    # now get size of padded region (buffer) at Model MPP
    buffer_right_l = int((w_l0 - (patch_n_w_l0 * p_s)) * mpp / MPP_MODEL_1)
    buffer_bottom_l = int((h_l0 - (patch_n_h_l0 * p_s)) * mpp / MPP_MODEL_1)
    # firstly bottom
    buffer_bottom = np.full((buffer_bottom_l, end_image.shape[1]), 0)
    temp_image = np.concatenate((end_image, buffer_bottom), axis=0)
    # now right side
    temp_image_he, temp_image_wi = temp_image.shape  # width and height
    buffer_right = np.full((temp_image_he, buffer_right_l), 0)
    end_image = np.concatenate((temp_image, buffer_right), axis=1).astype(np.uint8)

    end_image_1class = make_1class_map_thr(end_image, colors)
    end_image_1class = Image.fromarray(end_image_1class)
    end_image_1class = end_image_1class.resize((patch_n_w_l0*50, patch_n_h_l0*50), Image.Resampling.LANCZOS)

    return end_image_1class, end_image


def mask_to_geojson(mask_np, output_path, scale_factor=1.0):
    """
    Convert a semantic segmentation mask to GeoJSON with coordinate scaling

    Parameters:
    -----------
    mask_np : np.ndarray
        input PNG mask file
    output_path : str
        Path to save the output GeoJSON file
    scale_factor : float, optional, should be: model_mpp / slide_mpp
        Factor to scale coordinates by (default: 1.0)

    Returns:
    --------
    None
    """
    # Define class mapping
    CLASS_MAPPING = {
        1: "Normal Tissue",
        2: "Fold",
        3: "Darkspot & Foreign Object",
        4: "PenMarking",
        5: "Edge & Air Bubble",
        6: "OOF",  # Out of Focus
        7: "Background"
    }

    # Read the mask image
    mask = mask_np
    print("geojson 그리는 mask_np", mask_np.shape)

    # Dictionary to store features for each class
    features = []

    # Iterate through unique class values (1 to 7)
    for class_value in range(2, 7):
        # Create a binary mask for the current class
        class_mask = (mask == class_value).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Convert contours to GeoJSON features
        for contour in contours:
            # Flatten contour and reshape
            contour_points = contour.reshape(-1, 2)

            # Scale coordinates
            scaled_points = contour_points * scale_factor

            # Skip contours with less than 4 points
            if len(scaled_points) < 4:
                # print(f"Skipping contour with {len(scaled_points)} points for class {class_value}")
                continue

            # Ensure polygon is closed by adding first point at the end if needed
            polygon_points = scaled_points.tolist()
            if not np.array_equal(polygon_points[0], polygon_points[-1]):
                polygon_points.append(polygon_points[0])

            # Create feature for this polygon
            feature = {
                "type": "Feature",
                "properties": {
                    "class_id": int(class_value),
                    "classification": CLASS_MAPPING.get(class_value, "Unknown"),
                    "area": cv2.contourArea(contour) * (scale_factor ** 2)
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [polygon_points]
                }
            }

            features.append(feature)

    # Create GeoJSON structure
    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "class_mapping": CLASS_MAPPING,
            "scale_factor": scale_factor
        }
    }

    # Write to file
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)

    return geojson


def to_tensor_x(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_td_preprocessing(image,preprocessing_fn):
    image = np.array(image)
    x = preprocessing_fn(image)
    x = to_tensor_x(x)
    return x

def make_class_map (mask, class_colors):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, len(class_colors)):
        idx = mask == l
        r[idx] = class_colors [l][0]
        g[idx] = class_colors [l][1]
        b[idx] = class_colors [l][2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb