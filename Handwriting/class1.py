# import os
# import cv2
# from tqdm import tqdm

# # Set your image folder path
# image_folder = r'C:\Users\Luis Oliver\Datasets\validation_v2\\validation'

# # Get all image file names (e.g., only .jpg if needed)
# image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# # List to store failed image names
# failed_images = []

# # Loop through images with progress bar
# for file_name in tqdm(image_files, desc="Loading Images"):
#     file_path = os.path.join(image_folder, file_name)
#     image = cv2.imread(file_path)

#     if image is None:
#         failed_images.append(file_name)

# # Show results
# print("\n")
# if failed_images:
#     print("❌ Failed to load the following images:")
#     for f in failed_images:
#         print(" -", f)
# else:
#     print("✅ All images loaded successfully.")
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPUs detected:", tf.config.list_physical_devices('GPU'))