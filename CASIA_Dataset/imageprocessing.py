import os
import cv2
import numpy as np
from tqdm import tqdm


INPUT_DIR = "CASIA1"
OUTPUT_DIR = "CASIA1_segmented"
IMAGE_SIZE = (224, 224)


os.makedirs(OUTPUT_DIR, exist_ok=True)

def apply_clahe(img_gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_gray)

def preprocess_image(img_gray):
    clahe_applied = apply_clahe(img_gray)
    filtered = cv2.bilateralFilter(clahe_applied, d=9, sigmaColor=75, sigmaSpace=75)
    return filtered

def segment_iris(img_gray):
    enhanced = preprocess_image(img_gray)

    circles = cv2.HoughCircles(
        enhanced,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=40,
        param1=120,
        param2=25,
        minRadius=25,
        maxRadius=85
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = map(int, circles[0][0])

        mask = np.zeros_like(img_gray)
        cv2.circle(mask, (x, y), r, 255, -1)

        segmented = cv2.bitwise_and(enhanced, enhanced, mask=mask)

        h, w = img_gray.shape
        left = max(x - r, 0)
        right = min(x + r, w)
        top = max(y - r, 0)
        bottom = min(y + r, h)
        cropped = segmented[top:bottom, left:right]

        if cropped.size == 0:
            return None

        resized = cv2.resize(cropped, IMAGE_SIZE)
        return resized

    return None


for person_id in tqdm(os.listdir(INPUT_DIR), desc="Segmenting Persons"):
    person_path = os.path.join(INPUT_DIR, person_id)
    if not os.path.isdir(person_path):
        continue

    save_path = os.path.join(OUTPUT_DIR, person_id)
    os.makedirs(save_path, exist_ok=True)

    for img_file in os.listdir(person_path):
        img_path = os.path.join(person_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        segmented = segment_iris(img)
        if segmented is not None:
            out_path = os.path.join(save_path, img_file)
            cv2.imwrite(out_path, segmented)
        else:
            print(f"⚠️ Skipped (No iris found or error): {img_path}")

print("✅ High-level iris segmentation complete. Output folder:", OUTPUT_DIR)
