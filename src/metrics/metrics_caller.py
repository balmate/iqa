# with the defined methods in this file, metrics can be called in a simple way
import sewar.full_ref as sfr
import numpy as np

# ERGAS
def call_ergas(original_image: np.ndarray, deformed_image: np.ndarray, compared_to: str) -> str:
    return "----\nERGAS: original vs " + compared_to + ": " + str(sfr.ergas(original_image, deformed_image))

# MSE
def call_mse(original_image: np.ndarray, deformed_image: np.ndarray, compared_to: str) -> str:
    return "----\nMSE: original vs " + compared_to + ": " + str(sfr.mse(original_image, deformed_image))

# PSNR
def call_psnr(original_image: np.ndarray, deformed_image: np.ndarray, compared_to: str) -> str:
    return "----\nPSNR: original vs " + compared_to + ": " + str(sfr.psnr(original_image, deformed_image))

# SSIM
def call_ssim(original_image: np.ndarray, deformed_image: np.ndarray, compared_to: str) -> str:
    return "----\nSSIM: original vs " + compared_to + ": " + str(sfr.ssim(original_image, deformed_image))