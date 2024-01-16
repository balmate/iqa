import numpy as np
import utils.image_tools as image_tools
import metrics.metrics_caller as mc

def mse_comparison(original_image: np.ndarray, useNoise: bool = False, number_of_pixels_to_transform: int = 12000) -> None:
    if (useNoise):
        rotated_90_clockwise, rotated_90_counter_clockwise, rotated_180 = image_tools.generate_rotated_images_with_noise(original_image, number_of_pixels_to_transform)
        print("MSE with noised images:")
    else:
        rotated_90_clockwise, rotated_90_counter_clockwise, rotated_180 = image_tools.generate_rotated_images(original_image)
        print("MSE:")

    call_prints(original_image, rotated_90_clockwise, rotated_90_counter_clockwise, rotated_180, "mse")

def ergas_comparison(original_image: np.ndarray, useNoise: bool = False, number_of_pixels_to_transform: int = 12000) -> None:
    if (useNoise):
        rotated_90_clockwise, rotated_90_counter_clockwise, rotated_180 = image_tools.generate_rotated_images_with_noise(original_image, number_of_pixels_to_transform)
        print("ERGAS with noised images:")
    else:
        rotated_90_clockwise, rotated_90_counter_clockwise, rotated_180 = image_tools.generate_rotated_images(original_image)
        print("ERGAS:")

    call_prints(original_image, rotated_90_clockwise, rotated_90_counter_clockwise, rotated_180, "ergas")

def psnr_comparison(original_image: np.ndarray, useNoise: bool = False, number_of_pixels_to_transform: int = 12000) -> None:
    if (useNoise):
        rotated_90_clockwise, rotated_90_counter_clockwise, rotated_180 = image_tools.generate_rotated_images_with_noise(original_image, number_of_pixels_to_transform)
        print("PSNR with noised images:")
    else:
        rotated_90_clockwise, rotated_90_counter_clockwise, rotated_180 = image_tools.generate_rotated_images(original_image)
        print("PSNR:")

    call_prints(original_image, rotated_90_clockwise, rotated_90_counter_clockwise, rotated_180, "psnr")

def ssim_comparison(original_image: np.ndarray, useNoise: bool = False, number_of_pixels_to_transform: int = 12000) -> None:
    if (useNoise):
        rotated_90_clockwise, rotated_90_counter_clockwise, rotated_180 = image_tools.generate_rotated_images_with_noise(original_image, number_of_pixels_to_transform)
        print("SSIM with noised images:")
    else:
        rotated_90_clockwise, rotated_90_counter_clockwise, rotated_180 = image_tools.generate_rotated_images(original_image)
        print("SSIM:")

    call_prints(original_image, rotated_90_clockwise, rotated_90_counter_clockwise, rotated_180, "ssim")

def call_prints(original_image: np.ndarray, rotated_90_clockwise: np.ndarray, rotated_90_counter_clockwise: np.ndarray, rotated_180: np.ndarray, metric: str) -> None:
    if metric == "mse":
        print(mc.call_mse(original_image, original_image, "original"))
        print(mc.call_mse(original_image, image_tools.resize_to_original(rotated_90_clockwise), "90clockwise"))
        print(mc.call_mse(original_image, image_tools.resize_to_original(rotated_90_counter_clockwise), "90counterclockwise"))
        print(mc.call_mse(original_image, rotated_180, "180"))
        print("\n")
    elif metric == "ergas":
        print(mc.call_ergas(original_image, original_image, "original"))
        print(mc.call_ergas(original_image, image_tools.resize_to_original(rotated_90_clockwise), "90clockwise"))
        print(mc.call_ergas(original_image, image_tools.resize_to_original(rotated_90_counter_clockwise), "90counterclockwise"))
        print(mc.call_ergas(original_image, rotated_180, "180"))
        print("\n")
    elif metric == "psnr":
        print(mc.call_psnr(original_image, original_image, "original"))
        print(mc.call_psnr(original_image, image_tools.resize_to_original(rotated_90_clockwise), "90clockwise"))
        print(mc.call_psnr(original_image, image_tools.resize_to_original(rotated_90_counter_clockwise), "90counterclockwise"))
        print(mc.call_psnr(original_image, rotated_180, "180"))
        print("\n")
    elif metric == "ssim":
        print(mc.call_ssim(original_image, original_image, "original"))
        print(mc.call_ssim(original_image, image_tools.resize_to_original(rotated_90_clockwise), "90clockwise"))
        print(mc.call_ssim(original_image, image_tools.resize_to_original(rotated_90_counter_clockwise), "90counterclockwise"))
        print(mc.call_ssim(original_image, rotated_180, "180"))
        print("\n")