"""
==============================================================================
AI Vision Screening: Ocular Image Refinement & Feature Extraction Engine
==============================================================================
Processes uploaded raw images (face selfies, webcam frames, or ocular photos):
1. Ocular Region Localization & Adaptive ROI Cropping
2. Illumination Normalization & CLAHE Contrast Enhancement
3. Unsharp Masking & Focal Edge Accentuating
4. Clinical Ophthalmological Feature Extraction:
   - Focal Sharpness Score (Laplacian variance)
   - Meridional Curvature Asymmetry Index (Astigmatism power ratio)
   - Pupil-to-Iris Contrast Dynamics
   - Corneal Light Reflex (Hirschberg proxy) Symmetry
==============================================================================
"""

import io
import base64
import numpy as np
import cv2
from PIL import Image


def refine_ocular_image(pil_image: Image.Image) -> tuple[Image.Image, dict]:
    """
    Refines an input PIL image into an optimal, contrast-enhanced ocular crop
    and extracts quantitative physical features.
    
    Returns:
        (refined_pil_image, extracted_features_dict)
    """
    # 1. Convert PIL to RGB numpy array
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    img_rgb = np.array(pil_image)
    h, w = img_rgb.shape[:2]

    # 2. Ocular ROI Detection & Centering
    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    skin_area_ratio = float(np.mean(skin_mask > 0))

    cropped_rgb = img_rgb
    roi_detected = False

    # If significant skin is present and image is larger than standard eye crops
    if skin_area_ratio > 0.30 and max(h, w) > 260:
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_c = max(contours, key=cv2.contourArea)
            fx, fy, fw, fh = cv2.boundingRect(largest_c)
            if fw > 100 and fh > 100:
                ey1 = int(fy + fh * 0.22)
                ey2 = int(fy + fh * 0.55)
                ex1 = int(fx + fw * 0.15)
                ex2 = int(fx + fw * 0.85)

                ey1 = max(0, ey1)
                ey2 = min(h, ey2)
                ex1 = max(0, ex1)
                ex2 = min(w, ex2)

                eye_band = img_rgb[ey1:ey2, ex1:ex2]
                if eye_band.size > 0:
                    gray_band = cv2.cvtColor(eye_band, cv2.COLOR_RGB2GRAY)
                    blurred_band = cv2.GaussianBlur(gray_band, (9, 9), 0)
                    min_val, _, min_loc, _ = cv2.minMaxLoc(blurred_band)
                    
                    pupil_x = ex1 + min_loc[0]
                    pupil_y = ey1 + min_loc[1]

                    crop_dim = int(min(fw, fh) * 0.42)
                    cx1 = max(0, pupil_x - crop_dim // 2)
                    cy1 = max(0, pupil_y - crop_dim // 2)
                    cx2 = min(w, cx1 + crop_dim)
                    cy2 = min(h, cy1 + crop_dim)

                    if (cx2 - cx1) > 60 and (cy2 - cy1) > 60:
                        cropped_rgb = img_rgb[cy1:cy2, cx1:cx2]
                        roi_detected = True

    # 3. Illumination Normalization & CLAHE Contrast Enhancement
    lab = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    illum_normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # 4. Unsharp Masking for Limbal & Corneal Edge Preservation
    gaussian = cv2.GaussianBlur(illum_normalized, (0, 0), sigmaX=2.0)
    sharpened = cv2.addWeighted(illum_normalized, 1.25, gaussian, -0.25, 0)
    refined_rgb = np.clip(sharpened, 0, 255).astype(np.uint8)

    # 5. Quantitative Ocular Optical Feature Extraction
    gray = cv2.cvtColor(refined_rgb, cv2.COLOR_RGB2GRAY)
    
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mean_gx = float(np.mean(np.abs(sobel_x)))
    mean_gy = float(np.mean(np.abs(sobel_y)))
    asymmetry_ratio = round(mean_gx / (mean_gy + 1e-6), 3)

    mid_h, mid_w = gray.shape[:2]
    center_roi = gray[mid_h // 4 : 3 * mid_h // 4, mid_w // 4 : 3 * mid_w // 4]
    p_min = float(np.percentile(center_roi, 5))
    p_max = float(np.percentile(center_roi, 95))
    contrast_ratio = round((p_max - p_min) / (p_max + p_min + 1e-6), 3)

    left_half = gray[:, : mid_w // 2]
    right_half = cv2.flip(gray[:, mid_w // 2 :], 1)
    min_w_half = min(left_half.shape[1], right_half.shape[1])
    diff = np.abs(left_half[:, :min_w_half].astype(float) - right_half[:, :min_w_half].astype(float))
    reflex_symmetry_score = round(max(0.0, min(100.0, 100.0 - float(np.mean(diff)))), 2)

    extracted_features = {
        "roi_localized": roi_detected,
        "input_resolution": f"{w}x{h}",
        "refined_resolution": f"{refined_rgb.shape[1]}x{refined_rgb.shape[0]}",
        "focal_sharpness_score": round(lap_var, 2),
        "meridional_asymmetry_index": asymmetry_ratio,
        "pupil_iris_contrast_ratio": contrast_ratio,
        "reflex_symmetry_score": reflex_symmetry_score,
        "preprocessing": "Ocular ROI Localization + CLAHE Normalization + Unsharp Masking"
    }

    refined_pil = Image.fromarray(refined_rgb)
    return refined_pil, extracted_features


def extract_optical_feature_tensor(pil_image: Image.Image) -> np.ndarray:
    """
    Extracts a 4-dimensional normalized optical feature vector:
    [norm_sharpness, asymmetry_ratio, contrast_ratio, norm_reflex_symmetry]
    for direct fusion into the neural network classifier.
    """
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    gray = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
    
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    norm_sharpness = min(1.0, np.log1p(lap_var) / 10.0)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mean_gx = float(np.mean(np.abs(sobel_x)))
    mean_gy = float(np.mean(np.abs(sobel_y)))
    asymmetry_ratio = min(2.0, mean_gx / (mean_gy + 1e-6)) / 2.0

    mid_h, mid_w = gray.shape[:2]
    center_roi = gray[mid_h // 4 : 3 * mid_h // 4, mid_w // 4 : 3 * mid_w // 4]
    p_min = float(np.percentile(center_roi, 5))
    p_max = float(np.percentile(center_roi, 95))
    contrast_ratio = min(1.0, (p_max - p_min) / (p_max + p_min + 1e-6))

    left_half = gray[:, : mid_w // 2]
    right_half = cv2.flip(gray[:, mid_w // 2 :], 1)
    min_w_half = min(left_half.shape[1], right_half.shape[1])
    diff = np.abs(left_half[:, :min_w_half].astype(float) - right_half[:, :min_w_half].astype(float))
    norm_reflex_symmetry = max(0.0, min(1.0, 1.0 - float(np.mean(diff)) / 100.0))

    return np.array([norm_sharpness, asymmetry_ratio, contrast_ratio, norm_reflex_symmetry], dtype=np.float32)


def pil_to_base64_thumbnail(pil_image: Image.Image, max_size=(160, 160)) -> str:
    """Converts a PIL image to a base64 encoded data URI for UI display."""
    thumb = pil_image.copy()
    thumb.thumbnail(max_size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=85)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"
