import cv2
import numpy as np
import utils.noise_tools as noise_tools

def load_image(image_to_load: str) -> np.ndarray:
    original = cv2.imread(image_to_load)
    global original_image
    original_image = original
    return original

def generate_rotated_images(image_to_rotate: np.ndarray) -> np.ndarray:
    return cv2.rotate(image_to_rotate, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(image_to_rotate, cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.rotate(image_to_rotate, cv2.ROTATE_180)

def generate_rotated_images_with_noise(image_to_rotate: np.ndarray, number_of_pixels_to_transform: int):
    noisy_image = noise_tools.salt_and_pepper(image_to_rotate, number_of_pixels_to_transform)
    return cv2.rotate(noisy_image, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(noisy_image, cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.rotate(noisy_image, cv2.ROTATE_180)

def resize_to_original(image_to_resize: np.ndarray) -> np.ndarray:
    return cv2.resize(image_to_resize, (original_image.shape[1], original_image.shape[0]))

def show_image(window_title: str, image: np.ndarray) -> None:
    cv2.imshow(window_title, image)
    cv2.waitKey(0)