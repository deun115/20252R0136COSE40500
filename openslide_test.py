"""
SVS WSI: OpenSlide vs cuCIM 속도 비교 스크립트
- L0 전체 이미지 RAM 로드 시간
- 512x512 패치 추출 + 2D(그레이스케일) 변환 시간
- 동일 패치 좌표를 두 라이브러리에 공통 적용
"""
import argparse
import math
import time
import random
from typing import List, Tuple, Optional

import numpy as np

import openslide
from cucim import CuImage


# -----------------------------
# 공통 유틸
# -----------------------------
def to_gray_2d(arr: np.ndarray) -> np.ndarray:
    """
    입력: HxWxC (RGB/RGBA/ARGB 등) 또는 HxW (이미 그레이)
    출력: HxW (uint8) 그레이스케일
    """
    if arr.ndim == 2:
        # 이미 그레이
        g = arr
    else:
        # 채널 정리: RGBA -> RGB, ARGB -> RGB 등
        # 일반적으로 OpenSlide/ cuCIM 둘 다 RGBA 또는 RGB를 반환
        if arr.shape[2] == 4:
            rgb = arr[:, :, :3]
        elif arr.shape[2] >= 3:
            rgb = arr[:, :, :3]
        else:
            # 채널이 1개라면 그대로
            g = arr[:, :, 0]
            return g.astype(np.uint8) if g.dtype != np.uint8 else g

        # float로 변환 후, 표준 가중치로 그레이스케일
        rgbf = rgb.astype(np.float32)
        g = 0.299 * rgbf[:, :, 0] + 0.587 * rgbf[:, :, 1] + 0.114 * rgbf[:, :, 2]

    # 클리핑 및 uint8 변환
    g = np.clip(g, 0, 255)
    return g.astype(np.uint8)


def make_patch_coords(
    width: int,
    height: int,
    patch: int = 512,
    seed: int = 42,
    mode: str = "grid",  # "grid" 또는 "random"
) -> List[Tuple[int, int]]:
    """
    패치 좌표 (x, y) 리스트 생성. (L0 기준 좌표)
    """
    stride = patch

    coords = []
    if mode == "grid":
        nx = max(1, (width - patch) // stride + 1)
        ny = max(1, (height - patch) // stride + 1)
        for j in range(ny):
            y = j * stride
            if y + patch > height:
                y = height - patch
            for i in range(nx):
                x = i * stride
                if x + patch > width:
                    x = width - patch
                coords.append((x, y))
        # 중복 제거
        coords = list(dict.fromkeys(coords))

    else:
        # random
        random.seed(seed)
        n = 1024
        for _ in range(n):
            x = random.randint(0, max(0, width - patch))
            y = random.randint(0, max(0, height - patch))
            coords.append((x, y))

    return coords


# -----------------------------
# OpenSlide 파이프라인
# -----------------------------
def run_openslide(
    svs_path: str,
    patch: int = 512,
    mode: str = "grid",
    skip_full_load: bool = False,
):
    slide = openslide.OpenSlide(svs_path)
    w, h = slide.level_dimensions[0]

    timings = {}

    # L0 전체 로드 (RGBA PIL.Image 반환 -> np.array)
    full_img_np = None
    if not skip_full_load:
        t0 = time.perf_counter()
        pil_img = slide.read_region((0, 0), 0, (w, h))  # RGBA
        full_img_np = np.array(pil_img)  # HxWx4
        timings["full_load_sec"] = time.perf_counter() - t0
    else:
        timings["full_load_sec"] = None

    # 패치 좌표 생성(공통)
    coords = make_patch_coords(w, h, patch=patch, mode=mode)

    # 패치 추출 + 2D 변환
    t1 = time.perf_counter()
    grays = []
    for (x, y) in coords:
        if full_img_np is not None:
            # 메모리에서 직접 슬라이싱
            patch_arr = full_img_np[y : y + patch, x : x + patch]
        else:
            # 디스크에서 패치 단위 추출
            patch_arr = np.array(slide.read_region((x, y), 0, (patch, patch)))
        g = to_gray_2d(patch_arr)
        grays.append(g)
    timings["patch_extract_and_gray_sec"] = time.perf_counter() - t1

    # 결과 샘플 반환(메모리 사용 과다 방지)
    return {
        "lib": "OpenSlide",
        "width": w,
        "height": h,
        "num_patches": len(coords),
        "patch_size": patch,
        "timings": timings,
    }


# -----------------------------
# 메인
# -----------------------------
def main():
    svs_path = '/data/tcga_brca/TCGA-BH-A0DT-01Z-00-DX1.73AFCEBB-06B1-4870-ADA2-881511B1BE2D.svs'
    patch_size = 512
    mode = 'grid'
    skip_full_load = True


    results = []


    # cuCIM
    try:
        res_cu = run_cucim(
            svs_path,
            patch=patch_size,
            mode=mode,
            skip_full_load=skip_full_load,
        )
        results.append(res_cu)
    except Exception as e:
        print(f"[cuCIM] 오류: {e}")
        import traceback
        print(traceback.format_exc())


    # OpenSlide
    try:
        res_os = run_openslide(
            svs_path,
            patch=patch_size,
            mode=mode,
            skip_full_load=skip_full_load,
        )
        results.append(res_os)
    except Exception as e:
        print(f"[OpenSlide] 오류: {e}")


    # 결과 출력
    print("\n==== 결과 ====")
    for r in results:
        print(f"- 라이브러리: {r['lib']}")
        print(f"  이미지 크기 (L0): {r['width']} x {r['height']}")
        print(f"  패치 크기: {r['patch_size']}  | 패치 수: {r['num_patches']}")
        print(f"  전체 로드 시간: {r['timings']['full_load_sec']:.4f}s" if r['timings']['full_load_sec'] is not None else "  전체 로드: (건너뜀)")
        print(f"  패치 추출+그레이 변환: {r['timings']['patch_extract_and_gray_sec']:.4f}s")
        print()

    if not results:
        print("두 라이브러리 모두 실행 실패. 설치/환경을 확인하세요.")
