import cv2
import os


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
        else:
            print("invalid image in folder")
    return images


images = load_images_from_folder(
    "data/cube_64_R_4_random_angles/images")
l = len(images)
pass
