import os
import cv2
import numpy as np

# Paths (Change accordingly)
DATASET_PATH = r"E:\College\BDA\Project\BIG Data\New folder\MMU-Iris-Database"
PROCESSED_PATH = r"E:\College\BDA\Project\BIG Data\New folder\MMU-Iris-Processed"

IMG_SIZE = 128  # Resized image dimensions

def preprocess_and_save(dataset_path, save_path, img_size=IMG_SIZE):
    """Preprocesses iris images and saves them in a new folder."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for person_id in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_id)

        if os.path.isdir(person_path):
            for eye_side in ["left", "right"]:
                eye_path = os.path.join(person_path, eye_side)
                save_eye_path = os.path.join(save_path, person_id, eye_side)

                if os.path.exists(eye_path):
                    os.makedirs(save_eye_path, exist_ok=True)

                    for file in os.listdir(eye_path):
                        if file.endswith(".bmp"):  # Process only BMP images
                            img_path = os.path.join(eye_path, file)
                            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
                            img = cv2.resize(img, (img_size, img_size))  # Resize
                            save_img_path = os.path.join(save_eye_path, file)

                            cv2.imwrite(save_img_path, img)  # Save processed image

    print("Preprocessing complete. Processed images saved at:", save_path)

# Run the preprocessing
preprocess_and_save(DATASET_PATH, PROCESSED_PATH)
