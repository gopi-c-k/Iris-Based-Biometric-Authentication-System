# import os
# import cv2
# import numpy as np
# from tqdm import tqdm
#
# # Constants
# SOURCE_DIR = "CASIA1"
# DEST_DIR = "Processed"
# IMAGE_SIZE = (224, 224)
#
#
# def create_destination_folder_structure():
#     if not os.path.exists(DEST_DIR):
#         os.makedirs(DEST_DIR)
#     for person in os.listdir(SOURCE_DIR):
#         src_path = os.path.join(SOURCE_DIR, person)
#         dest_path = os.path.join(DEST_DIR, person)
#         if os.path.isdir(src_path) and not os.path.exists(dest_path):
#             os.makedirs(dest_path)
#
#
# def preprocess_iris_image(img_gray):
#     # 1. CLAHE for contrast enhancement
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     enhanced = clahe.apply(img_gray)
#
#     # 2. Bilateral filter for noise reduction
#     smoothed = cv2.bilateralFilter(enhanced, 9, 75, 75)
#
#     # 3. Otsu's thresholding (inverse to focus on dark iris)
#     _, thresh = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#
#     # 4. Morphological operations to refine mask
#     kernel = np.ones((3, 3), np.uint8)
#     refined_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#
#     return refined_mask
#
#
# def resize_and_normalize_image(img):
#     # Resize to 224x224 and normalize pixels to [0, 1]
#     img_resized = cv2.resize(img, IMAGE_SIZE)
#     img_normalized = img_resized.astype(np.float32) / 255.0
#     return img_normalized
#
#
# def process_image(image_path):
#     # Load image
#     img = cv2.imread(image_path)
#
#     # Convert to grayscale
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # Preprocess iris image
#     preprocessed_image = preprocess_iris_image(img_gray)
#
#     # Resize and normalize the image
#     final_image = resize_and_normalize_image(preprocessed_image)
#
#     return final_image
#
#
# def save_preprocessed_image(final_image, dest_path):
#     # Save the preprocessed image
#     cv2.imwrite(dest_path, (final_image * 255).astype(np.uint8))
#
#
# def process_images():
#     create_destination_folder_structure()
#
#     for person in tqdm(os.listdir(SOURCE_DIR)):
#         person_src_path = os.path.join(SOURCE_DIR, person)
#         person_dest_path = os.path.join(DEST_DIR, person)
#
#         if os.path.isdir(person_src_path):
#             for img_name in os.listdir(person_src_path):
#                 img_path = os.path.join(person_src_path, img_name)
#
#                 # Process each image
#                 final_image = process_image(img_path)
#
#                 # Define destination path
#                 dest_img_path = os.path.join(person_dest_path, img_name)
#
#                 # Save the preprocessed image
#                 save_preprocessed_image(final_image, dest_img_path)
#
#
# if __name__ == "__main__":
#     process_images()


#

import os
import cv2
import numpy as np
from tqdm import tqdm

# Paths
INPUT_DIR = "CASIA1"
OUTPUT_DIR = "CASIA1_segmented"
IMAGE_SIZE = (224, 224)

# Create output folder
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

    # Detect iris using Hough Circles
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

# Processing loop
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
