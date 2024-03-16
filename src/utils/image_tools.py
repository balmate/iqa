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
                                     gamma: float = 0.5, blur: list = (5,5), fade_percent: float = 0.2, saturation: float = 0.2, alpha: float = 0.5, zoom: float = 1.5):
    noisy_image = create_concrete_noisy_image(image_to_modify, noise_type, number_of_pixels_to_transform, mean, sigma, gamma, blur, fade_percent, saturation, alpha, zoom)
    return cv2.rotate(noisy_image, cv2.ROTATE_180)

def resize_to_original(image_to_resize: np.ndarray) -> np.ndarray:
    return cv2.resize(image_to_resize, (original_image.shape[1], original_image.shape[0]))

def show_image(window_title: str, image: np.ndarray) -> None:
    return
    cv2.imshow(window_title, image)
    cv2.waitKey(0)

def create_scaled_image(original_image: np.ndarray, scale_value: int):
    new_size = (int(original_image.shape[1] * scale_value / 100), int(original_image.shape[0] * scale_value / 100))
    return cv2.resize(original_image, new_size, interpolation=cv2.INTER_AREA)

def create_concrete_noisy_image(image_to_return: np.ndarray, noise_type: str, number_of_pixels_to_transform: int, mean: float, sigma: float, gamma: float, blur: list,
                                 fade_percent: float, saturation: float, alpha: float, zoom: float):
    if noise_type == "salt&pepper":
        return noise_tools.salt_and_pepper(image_to_return, number_of_pixels_to_transform)
    elif noise_type == "gaussian":
        return noise_tools.gaussian(image_to_return, mean, sigma)
    elif noise_type == "poisson":
        return noise_tools.poisson(image_to_return, gamma)
    elif noise_type == "blur":
        return noise_tools.blur(image_to_return, blur)
    elif noise_type == "fade":
        return noise_tools.fade(image_to_return, fade_percent)
    elif noise_type == "saturation":
        return noise_tools.saturation(image_to_return, saturation)
    elif noise_type == "contrast":
        return noise_tools.contrast(image_to_return, alpha)
    elif noise_type == "zoom":
        return noise_tools.zoom(image_to_return, zoom)
    
def create_rotated_image(image: np.ndarray, angle: int) -> np.ndarray:
  return imutils.rotate(image, angle)

def save_image(image: np.ndarray, path: str) -> None:
    print("path: " + path)
    cv2.imwrite(path, image)

# noise combos
def blur_with_salt_and_pepper(image: np.ndarray, kernel: list, pixels_to_transform: int, file_name: str) -> np.ndarray:
    blurred = noise_tools.blur(image, kernel)
    combo = noise_tools.salt_and_pepper(blurred, pixels_to_transform)
    save_image(combo, file_name)
    return combo

def rotated_with_salt_and_pepper(image: np.ndarray, angle: int, pixels_to_transform: int, file_name: str) -> np.ndarray:
    rotated = create_rotated_image(image, angle)
    combo = noise_tools.salt_and_pepper(rotated, pixels_to_transform)
    save_image(combo, file_name)
    return combo

def rotated_with_blur(image: np.ndarray, angle: int, kernel: list, file_name: str) -> np.ndarray:
    rotated = create_rotated_image(image, angle)
    combo = noise_tools.blur(rotated, kernel)
    save_image(combo, file_name)
    return combo

def rotated_with_contrast(image: np.ndarray, angle: int, alpha: float, file_name: str) -> np.ndarray:
    rotated = create_rotated_image(image, angle)
    combo = noise_tools.contrast(rotated, alpha)
    save_image(combo, file_name)
    return combo

def saturation_with_salt_and_pepper(image: np.ndarray, percent: float, pixels_to_transform: int, file_name: str) -> np.ndarray:
    saturated = noise_tools.saturation(image, percent)
    combo = noise_tools.salt_and_pepper(saturated, pixels_to_transform)
    save_image(combo, file_name)
    return combo

def fade_with_contrast(image: np.ndarray, percent: float, alpha: float, file_name: str) -> np.ndarray:
    faded = noise_tools.fade(image, percent)
    combo = noise_tools.contrast(faded, alpha)
    save_image(combo, file_name)
    return combo