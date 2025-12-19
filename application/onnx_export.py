import torch
import segmentation_models_pytorch as smp


def load_td_model(MODEL_TD_PATH, DEVICE="cuda"):
    # MODEL TISSUE DETECTION:
    ENCODER_MODEL_TD = 'timm-efficientnet-b0'
    ENCODER_MODEL_TD_WEIGHTS = 'imagenet'

    model = smp.UnetPlusPlus(
        encoder_name=ENCODER_MODEL_TD,
        encoder_weights=ENCODER_MODEL_TD_WEIGHTS,
        classes=2,
        activation=None,
    )

    model.load_state_dict(torch.load(MODEL_TD_PATH, map_location='cpu'))
    model.to(DEVICE)
    return model

def load_ad_model(MODEL_QC_PATH, map_location="cuda", weights_only=False):
    model_prim = torch.load(MODEL_QC_PATH, map_location=map_location, weights_only=False)

    return model_prim

import onnxruntime as ort

def load_td_onnx_session(
    ENCODER_MODEL_TD: str,
    ENCODER_MODEL_TD_WEIGHTS: str,
    MODEL_TD_ONNX_PATH: str,
    use_gpu: bool = True,
):
    """
    ONNX로 export된 TD 모델을 로드해서 onnxruntime 세션을 반환.
    기존 load_td_model과 비슷하게 preprocessing_fn도 함께 반환.
    """
    # smp preprocessing 함수는 그대로 사용
    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        ENCODER_MODEL_TD, ENCODER_MODEL_TD_WEIGHTS
    )

    # ONNX Runtime providers 설정 (GPU 우선, 안 되면 CPU fallback)
    if use_gpu:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(MODEL_TD_ONNX_PATH, providers=providers)

    return preprocessing_fn, session


def load_sd_onnx_session(MODEL_AD_ONNX_PATH, use_gpu=True):
    # ONNX Runtime providers 설정 (GPU 우선, 안 되면 CPU fallback)
    if use_gpu:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(MODEL_AD_ONNX_PATH, providers=providers)

    return session


if __name__ == "__main__":
    td_weight_path = "/workspace/01_WSI_inference_OPENSLIDE_QC/models/td/Tissue_Detection_MPP10.pth"

    ENCODER_MODEL_TD = 'timm-efficientnet-b0'
    ENCODER_MODEL_TD_WEIGHTS = 'imagenet'
    onnx_path = "/workspace/application/tissue_detection.onnx"

    # td_model = load_td_model(td_weight_path)

    # input_tensor = torch.randn([1, 3, 512, 512], device="cuda")
    # torch.onnx.export(
    #     td_model, 
    #     input_tensor,
    #     "/workspace/application/tissue_detection.onnx"
    # )
    preprocessing_fn, session = load_td_onnx_session(ENCODER_MODEL_TD, ENCODER_MODEL_TD_WEIGHTS, onnx_path)
    print("onnx 모델 로드 성공")
    print(session)

    ad_weight_path = '/workspace/01_WSI_inference_OPENSLIDE_QC/models/qc/GrandQC_MPP1.pth'
    ENCODER_MODEL = 'timm-efficientnet-b0'
    ENCODER_MODEL_WEIGHTS = 'imagenet'
    onnx_path = "/workspace/application/artifact_detection.onnx"

    ad_model = load_ad_model(ad_weight_path, map_location="cuda")
    input_tensor = torch.randn([1, 3, 512, 512], device="cuda")
    torch.onnx.export(
        ad_model, 
        input_tensor,
        "/workspace/application/artifact_detection.onnx",
        input_names=["input"],
    )
