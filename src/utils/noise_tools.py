import cv2
import numpy as np
import random
import utils.image_tools as image_tools

def salt_and_pepper(image: np.ndarray, number_of_pixels_to_transform: int) -> np.ndarray:
    '''
    Add salt-pepper noise to the image based on the number_of_pixels_to_transform parameter.
    '''
    # image dimensions
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
    call_save(image_to_return, "kadid_transforms/salt&pepper/", f"salt&pepper_{number_of_pixels_to_transform}")
    return image_to_return

def gaussian(image: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    '''
    Add gaussian noise to the image based on the incoming params.
    '''
    noise = np.zeros(image.shape, np.uint8)
    # add normally distributed noise
    cv2.randn(noise, mean, sigma)

    # add noise to the image
    image_to_return = cv2.add(image, cv2.add(image, noise))
    # save
    call_save(image_to_return, "kadid_transforms/gaussian/", f"gaussian_m{mean}_s_{sigma}")
    return image_to_return

def poisson(image: np.ndarray, gamma: float) -> np.ndarray:
    '''
    Add poisson noise to the image based on gamma parameter.
    '''
    # create poisson distribution, then "scale" back the data
    noisy = np.random.poisson(image * 255 * gamma)
    noisy = np.clip(noisy, 0, 255)

    # use as smaller datatype in the returned array
    image_to_return = noisy.astype(np.uint8)
    # save
    call_save(image_to_return, "kadid_transforms/poisson/", f"poisson_{gamma}")
    return image_to_return

def blur(image: np.ndarray, kernel: list) -> np.ndarray:
    '''
    Add blur noise to the image based on the incoming kernel param.
    '''
    # use the cv2 built in function
    image_to_return = cv2.blur(image, kernel)
    # save
    call_save(image_to_return, "kadid_transforms/blur/", f"blur_{kernel}")
    return image_to_return

def fade(image: np.ndarray, percent: float) -> np.ndarray:
    '''
    Add fade noise to the image based on the incoming percentage.
    '''
    # use the percent param to "scale" down the intenzities of the image
    image_to_return = np.clip(image * (1 - percent) + 255 * percent, 0, 255).astype(np.uint8)
    # save
    call_save(image_to_return, "kadid_transforms/fade/", f"fade_{percent}")
    return image_to_return

def saturation(image: np.ndarray, percent: float) -> np.ndarray:
    '''
    Add saturation noise to the image based on the incoming percent.
    '''
    # transform the image to HSV color space to get the saturation channel and modify it
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * percent, 0, 255)
    # transform the image back to bgr
    image_to_return = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    # save
    call_save(image_to_return, "kadid_transforms/saturation/", f"saturation_{percent}")
    return image_to_return

def contrast(image: np.ndarray, alpha: float) -> np.ndarray:
    '''
    Add contrast noise to the image based on the incoming alpha param.
    '''
    # use the cv2's built in function to increase/decrease the contrast of the image
    image_to_return = cv2.convertScaleAbs(image, alpha = alpha)
    # save
    call_save(image_to_return, "kadid_transforms/contrast/", f"contrast_{alpha}")
    return image_to_return

def zoom(image: np.ndarray, zoom_factor: np.ndarray, middle_coord: list = (306, 192)) -> np.ndarray:
    '''
    Create a zoomed copy of the image based on the incoming factor param. The function also got the middle cordinate of the image as default param (KADID image cases).
    '''
    # calculate middle coordinate if the param is none
    cy, cx = [ i/2 for i in image.shape[:-1] ] if middle_coord is None else middle_coord[::-1]
    
    # create rotation matrix, and transform the image matrix with the cv2 built in warpAffine method
    rot_mat = cv2.getRotationMatrix2D((cx,cy), 0, zoom_factor)
    image_to_return = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    # save
    call_save(image_to_return, "kadid_transforms/zoom/", f"zoom_{zoom_factor}")
    return image_to_return 

def pick_random_coordinate(row: int, col: int) -> int:
    '''
    Function to pick a random coordinate in the dimensions fetched as parameters.
    '''
    return random.randint(0, row - 1), random.randint(0, col - 1)

def call_save(image: np.ndarray, path: str, file_name: str) -> None:
    '''
    Function to save images to the specific folder of the project.
    '''
    file_name = image_tools.current_image_name + "_" + file_name.replace('.', '_') + ".jpg"
    image_tools.save_image(image, path + file_name)