import cv2
import numpy as np
import random

def salt_and_pepper(image: np.ndarray, number_of_pixels_to_transform: int) -> np.ndarray:
    row, col = image.shape[1], image.shape[0]

    # salt
    for _ in range(number_of_pixels_to_transform):
        x, y = pick_random_coordinate(row, col)
        image[y][x] = 255

    # pepper
    for _ in range(number_of_pixels_to_transform):
        x, y = pick_random_coordinate(row, col)
        image[y][x] = 0
    return image


def pick_random_coordinate(row: int, col: int) -> int:
    return random.randint(0, row - 1), random.randint(0, col - 1)