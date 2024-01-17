import numpy as np
import utils.image_tools as image_tools
import metrics.metrics_caller as mc

def mse_comparison(original_image: np.ndarray, useNoise: bool = False, number_of_pixels_to_transform: int = 20000) -> None:
    if (useNoise):
        rotated_90_clockwise, rotated_90_counter_clockwise, rotated_180 = image_tools.generate_rotated_images_with_noise(original_image, number_of_pixels_to_transform)
        print("MSE with noised images:")
    else:
        rotated_90_clockwise, rotated_90_counter_clockwise, rotated_180 = image_tools.generate_rotated_images(original_image)
        print("MSE:")

    call_prints(original_image, rotated_90_clockwise, rotated_90_counter_clockwise, rotated_180, "mse")

def ergas_comparison(original_image: np.ndarray, useNoise: bool = False, number_of_pixels_to_transform: int = 20000) -> None:
    if (useNoise):
        rotated_90_clockwise, rotated_90_counter_clockwise, rotated_180 = image_tools.generate_rotated_images_with_noise(original_image, number_of_pixels_to_transform)
        print("ERGAS with noised images:")
    else:
        rotated_90_clockwise, rotated_90_counter_clockwise, rotated_180 = image_tools.generate_rotated_images(original_image)
        print("ERGAS:")

    call_prints(original_image, rotated_90_clockwise, rotated_90_counter_clockwise, rotated_180, "ergas")

def psnr_comparison(original_image: np.ndarray, useNoise: bool = False, number_of_pixels_to_transform: int = 20000) -> None:
    if (useNoise):
        rotated_90_clockwise, rotated_90_counter_clockwise, rotated_180 = image_tools.generate_rotated_images_with_noise(original_image, number_of_pixels_to_transform)
        print("PSNR with noised images:")
    else:
        rotated_90_clockwise, rotated_90_counter_clockwise, rotated_180 = image_tools.generate_rotated_images(original_image)
        print("PSNR:")

    call_prints(original_image, rotated_90_clockwise, rotated_90_counter_clockwise, rotated_180, "psnr")

def ssim_comparison(original_image: np.ndarray, useNoise: bool = False, number_of_pixels_to_transform: int = 20000) -> None:
    if (useNoise):
        rotated_90_clockwise, rotated_90_counter_clockwise, rotated_180 = image_tools.generate_rotated_images_with_noise(original_image, number_of_pixels_to_transform)
        print("SSIM with noised images:")
    else:
        rotated_90_clockwise, rotated_90_counter_clockwise, rotated_180 = image_tools.generate_rotated_images(original_image)
        print("SSIM:")

    call_prints(original_image, rotated_90_clockwise, rotated_90_counter_clockwise, rotated_180, "ssim")

def msssim_comparison(original_image: np.ndarray, useNoise: bool = False, number_of_pixels_to_transform: int = 20000) -> None:
    if (useNoise):
        rotated_90_clockwise, rotated_90_counter_clockwise, rotated_180 = image_tools.generate_rotated_images_with_noise(original_image, number_of_pixels_to_transform)
        print("MS-SSIM with noised images:")
    else:
        rotated_90_clockwise, rotated_90_counter_clockwise, rotated_180 = image_tools.generate_rotated_images(original_image)
        print("MS-SSIM:")

    call_prints(original_image, rotated_90_clockwise, rotated_90_counter_clockwise, rotated_180, "msssim")

def vif_comparison(original_image: np.ndarray, useNoise: bool = False, number_of_pixels_to_transform: int = 20000) -> None:
    if (useNoise):
        rotated_90_clockwise, rotated_90_counter_clockwise, rotated_180 = image_tools.generate_rotated_images_with_noise(original_image, number_of_pixels_to_transform)
        print("VIF with noised images:")
    else:
        rotated_90_clockwise, rotated_90_counter_clockwise, rotated_180 = image_tools.generate_rotated_images(original_image)
        print("VIF:")

    call_prints(original_image, rotated_90_clockwise, rotated_90_counter_clockwise, rotated_180, "vif")

def scc_comparison(original_image: np.ndarray, useNoise: bool = False, number_of_pixels_to_transform: int = 20000) -> None:
    if (useNoise):
        rotated_90_clockwise, rotated_90_counter_clockwise, rotated_180 = image_tools.generate_rotated_images_with_noise(original_image, number_of_pixels_to_transform)
        print("SCC with noised images:")
    else:
        rotated_90_clockwise, rotated_90_counter_clockwise, rotated_180 = image_tools.generate_rotated_images(original_image)
        print("SCC:")

    call_prints(original_image, rotated_90_clockwise, rotated_90_counter_clockwise, rotated_180, "scc")

def sam_comparison(original_image: np.ndarray, useNoise: bool = False, number_of_pixels_to_transform: int = 20000) -> None:
    if (useNoise):
        rotated_90_clockwise, rotated_90_counter_clockwise, rotated_180 = image_tools.generate_rotated_images_with_noise(original_image, number_of_pixels_to_transform)
        print("SAM with noised images:")
    else:
        rotated_90_clockwise, rotated_90_counter_clockwise, rotated_180 = image_tools.generate_rotated_images(original_image)
        print("SAM:")

    call_prints(original_image, rotated_90_clockwise, rotated_90_counter_clockwise, rotated_180, "sam")

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
    elif metric == "msssim":
        print(mc.call_msssim(original_image, original_image, "original"))
        print(mc.call_msssim(original_image, image_tools.resize_to_original(rotated_90_clockwise), "90clockwise"))
        print(mc.call_msssim(original_image, image_tools.resize_to_original(rotated_90_counter_clockwise), "90counterclockwise"))
        print(mc.call_msssim(original_image, rotated_180, "180"))
        print("\n")
    elif metric == "vif":
        print(mc.call_vif(original_image, original_image, "original"))
        print(mc.call_vif(original_image, image_tools.resize_to_original(rotated_90_clockwise), "90clockwise"))
        print(mc.call_vif(original_image, image_tools.resize_to_original(rotated_90_counter_clockwise), "90counterclockwise"))
        print(mc.call_vif(original_image, rotated_180, "180"))
        print("\n")
    elif metric == "scc":
        print(mc.call_scc(original_image, original_image, "original"))
        print(mc.call_scc(original_image, image_tools.resize_to_original(rotated_90_clockwise), "90clockwise"))
        print(mc.call_scc(original_image, image_tools.resize_to_original(rotated_90_counter_clockwise), "90counterclockwise"))
        print(mc.call_scc(original_image, rotated_180, "180"))
        print("\n")
    elif metric == "sam":
        print(mc.call_sam(original_image, original_image, "original"))
        print(mc.call_sam(original_image, image_tools.resize_to_original(rotated_90_clockwise), "90clockwise"))
        print(mc.call_sam(original_image, image_tools.resize_to_original(rotated_90_counter_clockwise), "90counterclockwise"))
        print(mc.call_sam(original_image, rotated_180, "180"))
        print("\n")