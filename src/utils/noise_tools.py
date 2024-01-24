import cv2
import numpy as np
import random

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
    return image_to_return

def gaussian(image: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    noise = np.zeros(image.shape, np.uint8)
    cv2.randn(noise, mean, sigma)

    return cv2.add(image, cv2.add(image, noise))

def poisson(image: np.ndarray, gamma: float) -> np.ndarray:
    noisy = np.random.poisson(image * 255 * gamma)
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

def pick_random_coordinate(row: int, col: int) -> int:
    return random.randint(0, row - 1), random.randint(0, col - 1)