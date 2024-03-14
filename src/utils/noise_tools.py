import cv2
import numpy as np
import random
import utils.image_tools as image_tools

def salt_and_pepper(image: np.ndarray, number_of_pixels_to_transform: int) -> np.ndarray:
    row, col = image.shape[1], image.shape[0]
    image_to_return = np.copy(image)

    # salt
    for _ in range(number_of_pixels_to_transform):
        x, y = pick_random_coordinate(row, col)
        image_to_return[y][x] = 255

    # pepper
    for _ in range(number_of_pixels_to_transform):
        x, y = pick_random_coordinate(row, col)
        image_to_return[y][x] = 0
    
    # save
    call_save(image_to_return, "transformed_images/salt&pepper/", f"salt&pepper_{number_of_pixels_to_transform}")
    return image_to_return

def gaussian(image: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    noise = np.zeros(image.shape, np.uint8)
    cv2.randn(noise, mean, sigma)

    image_to_return = cv2.add(image, cv2.add(image, noise))
    call_save(image_to_return, "transformed_images/gaussian/", f"gaussian_m{mean}_s_{sigma}")
    return image_to_return

def poisson(image: np.ndarray, gamma: float) -> np.ndarray:
    noisy = np.random.poisson(image * 255 * gamma)
    noisy = np.clip(noisy, 0, 255)

    image_to_return = noisy.astype(np.uint8)
    call_save(image_to_return, "transformed_images/poisson/", f"poisson_{gamma}")
    return image_to_return

def blur(image: np.ndarray, kernel: list) -> np.ndarray:
    image_to_return = cv2.blur(image, kernel)
    call_save(image_to_return, "transformed_images/blur/", f"blur_{kernel}")
    return image_to_return

def fade(image: np.ndarray, percent: float) -> np.ndarray:
    image_to_return = np.clip(image * (1 - percent) + 255 * percent, 0, 255).astype(np.uint8)
    call_save(image_to_return, "transformed_images/fade/", f"fade_{percent}")
    return image_to_return

def saturation(image: np.ndarray, percent: float) -> np.ndarray:
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * percent, 0, 255)
    image_to_return = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    call_save(image_to_return, "transformed_images/saturation/", f"saturation_{percent}")
    return image_to_return

def contrast(image: np.ndarray, alpha: float) -> np.ndarray:
    image_to_return = cv2.convertScaleAbs(image, alpha = alpha)
    call_save(image_to_return, "transformed_images/contrast/", f"contrast_{alpha}")
    return image_to_return

def zoom(image: np.ndarray, zoom_factor: np.ndarray, middle_coord: list = (306, 192)) -> np.ndarray:
    cy, cx = [ i/2 for i in image.shape[:-1] ] if middle_coord is None else middle_coord[::-1]
    
    rot_mat = cv2.getRotationMatrix2D((cx,cy), 0, zoom_factor)
    image_to_return = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    call_save(image_to_return, "transformed_images/zoom/", f"zoom_{zoom_factor}")
    return image_to_return 

def pick_random_coordinate(row: int, col: int) -> int:
    return random.randint(0, row - 1), random.randint(0, col - 1)

def call_save(image: np.ndarray, path: str, file_name: str) -> None:
    file_name = image_tools.current_image_name + "_" + file_name.replace('.', '_') + ".jpg"
    image_tools.save_image(image, path + file_name)