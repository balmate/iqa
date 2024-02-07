# with the defined methods in this file, metrics can be called in a simple way
import sewar.full_ref as sfr
import numpy as np
from decimal import Decimal, ROUND_HALF_UP

# MSE
def call_mse(original_image: np.ndarray, deformed_image: np.ndarray, compared_to: str) -> str:
    res = Decimal(sfr.mse(original_image, deformed_image))
    return "MSE: original vs " + compared_to + ": " + quantize_decimal(res)

# ERGAS
def call_ergas(original_image: np.ndarray, deformed_image: np.ndarray, compared_to: str) -> str:
    res = Decimal(sfr.ergas(original_image, deformed_image))
    return "ERGAS: original vs " + compared_to + ": " + quantize_decimal(res)

# PSNR
def call_psnr(original_image: np.ndarray, deformed_image: np.ndarray, compared_to: str) -> str:
    res = Decimal(sfr.psnr(original_image, deformed_image))
    return "PSNR: original vs " + compared_to + ": " + quantize_decimal(res)

# SSIM
def call_ssim(original_image: np.ndarray, deformed_image: np.ndarray, compared_to: str) -> str:
    return "SSIM: original vs " + compared_to + ": " + str(sfr.ssim(original_image, deformed_image))

# MS-SSIM
def call_msssim(original_image: np.ndarray, deformed_image: np.ndarray, compared_to: str) -> str:
    return "MS-SSIM: original vs " + compared_to + ": " + str(sfr.msssim(original_image, deformed_image))

# VIF
def call_vif(original_image: np.ndarray, deformed_image: np.ndarray, compared_to: str) -> str:
    res = Decimal(sfr.vifp(original_image, deformed_image))
    return "VIF: original vs " + compared_to + ": " + quantize_decimal(res)

# SCC
def call_scc(original_image: np.ndarray, deformed_image: np.ndarray, compared_to: str) -> str:
    res = Decimal(sfr.scc(original_image, deformed_image))
    return "SCC: original vs " + compared_to + ": " + quantize_decimal(res)

# SAM
def call_sam(original_image: np.ndarray, deformed_image: np.ndarray, compared_to: str) -> str:
    res = Decimal(sfr.sam(original_image, deformed_image))
    return "SAM: original vs " + compared_to + ": " + quantize_decimal(res)

# helper methods
def quantize_decimal(input: Decimal) -> str:
    return str(input.quantize(Decimal('0.001'), ROUND_HALF_UP))