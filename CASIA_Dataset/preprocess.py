import os
import cv2
import numpy as np
from tqdm import tqdm


INPUT_DIR = "CASIA1"
OUTPUT_DIR = "CASIA1_processed"
IMAGE_SIZE = (224, 224)
USE_CLAHE = True  

os.makedirs(OUTPUT_DIR, exist_ok=True)


clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if USE_CLAHE else None


for person_id in tqdm(os.listdir(INPUT_DIR), desc="Processing Persons"):
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

 
        img = cv2.resize(img, IMAGE_SIZE)

    
        if USE_CLAHE:
            img = clahe.apply(img)

        img = img.astype(np.float32) / 255.0
        img = (img * 255).astype(np.uint8)

        out_path = os.path.join(save_path, img_file)
        cv2.imwrite(out_path, img)

print("✅ All images processed and saved to:", OUTPUT_DIR)
