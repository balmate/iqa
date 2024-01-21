import cv2
import numpy as np
import utils.noise_tools as noise_tools

def load_image(image_to_load: str) -> np.ndarray:
    original = cv2.imread(image_to_load)
    global original_image
    original_image = original
    return original

def generate_180_rotated(image_to_rotate: np.ndarray) -> np.ndarray:
    return cv2.rotate(image_to_rotate, cv2.ROTATE_180)

def generate_180_rotated_with_noise(image_to_modify: np.ndarray, noiseType: str, number_of_pixels_to_transform: int, mean: float, sigma: float, gamma: float):
    noisy_image = create_concrete_noisy_image(image_to_modify, noiseType, number_of_pixels_to_transform, mean, sigma, gamma)
    return cv2.rotate(noisy_image, cv2.ROTATE_180)

def resize_to_original(image_to_resize: np.ndarray) -> np.ndarray:
    return cv2.resize(image_to_resize, (original_image.shape[1], original_image.shape[0]))

def show_image(window_title: str, image: np.ndarray) -> None:
    cv2.imshow(window_title, image)
    cv2.waitKey(0)

def create_concrete_noisy_image(image_to_return, noiseType: str, number_of_pixels_to_transform: int, mean: float, sigma: float, gamma: float):
    if noiseType == "salt&pepper":
        return noise_tools.salt_and_pepper(image_to_return, number_of_pixels_to_transform)
    elif noiseType == "gaussian":
        return noise_tools.gaussian(image_to_return, mean, sigma)
    elif noiseType == "poisson":
        return noise_tools.poisson(image_to_return, gamma)