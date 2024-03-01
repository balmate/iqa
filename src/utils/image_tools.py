import cv2
import numpy as np
import utils.noise_tools as noise_tools
import imutils

def load_image(image_to_load: str) -> np.ndarray:
    original = cv2.imread(image_to_load)
    global original_image
    original_image = original
    return original

def generate_180_rotated_with_noise(image_to_modify: np.ndarray, noise_type: str, number_of_pixels_to_transform: int = 15000, mean: float = 0.5, sigma: float = 100,
                                     gamma: float = 0.5, blur: list = (5,5)):
    noisy_image = create_concrete_noisy_image(image_to_modify, noise_type, number_of_pixels_to_transform, mean, sigma, gamma, blur)
    return cv2.rotate(noisy_image, cv2.ROTATE_180)

def resize_to_original(image_to_resize: np.ndarray) -> np.ndarray:
    return cv2.resize(image_to_resize, (original_image.shape[1], original_image.shape[0]))

def show_image(window_title: str, image: np.ndarray) -> None:
    cv2.imshow(window_title, image)
    cv2.waitKey(0)

def create_scaled_image(original_image: np.ndarray, scale_value: int):
    new_size = (int(original_image.shape[1] * scale_value / 100), int(original_image.shape[0] * scale_value / 100))
    return cv2.resize(original_image, new_size, interpolation=cv2.INTER_AREA)

def create_concrete_noisy_image(image_to_return, noise_type: str, number_of_pixels_to_transform: int, mean: float, sigma: float, gamma: float, blur: list):
    if noise_type == "salt&pepper":
        return noise_tools.salt_and_pepper(image_to_return, number_of_pixels_to_transform)
    elif noise_type == "gaussian":
        return noise_tools.gaussian(image_to_return, mean, sigma)
    elif noise_type == "poisson":
        return noise_tools.poisson(image_to_return, gamma)
    elif noise_type == "blur":
        return noise_tools.blur(image_to_return, blur)
    
def create_rotated_image(image: np.ndarray, angle: int) -> np.ndarray:
  return imutils.rotate(image, angle)