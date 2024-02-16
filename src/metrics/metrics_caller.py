# with the defined methods in this file, metrics can be called in a simple way
import sewar.full_ref as sfr
import numpy as np
from decimal import Decimal, ROUND_HALF_UP

# MSE
def call_mse(original_image: np.ndarray, deformed_image: np.ndarray, compared_to: str):
    res = sfr.mse(original_image, deformed_image)
    print("MSE: original vs " + compared_to + ": " + quantize_decimal(Decimal(res)))
    return res

# ERGAS
def call_ergas(original_image: np.ndarray, deformed_image: np.ndarray, compared_to: str):
    res = sfr.ergas(original_image, deformed_image)
    print("ERGAS: original vs " + compared_to + ": " + quantize_decimal(Decimal(res)))
    return res

# PSNR
def call_psnr(original_image: np.ndarray, deformed_image: np.ndarray, compared_to: str):
    res = sfr.psnr(original_image, deformed_image)
    print("PSNR: original vs " + compared_to + ": " + quantize_decimal(Decimal(res)))
    return res

# SSIM
def call_ssim(original_image: np.ndarray, deformed_image: np.ndarray, compared_to: str):
    res = sfr.ssim(original_image, deformed_image)
    print("SSIM: original vs " + compared_to + ": " + str(res))
    return res

# MS-SSIM
def call_msssim(original_image: np.ndarray, deformed_image: np.ndarray, compared_to: str):
    res = sfr.msssim(original_image, deformed_image)
    print("MS-SSIM: original vs " + compared_to + ": " + str(res))
    return res

# VIF
def call_vif(original_image: np.ndarray, deformed_image: np.ndarray, compared_to: str):
    res = sfr.vifp(original_image, deformed_image)
    print("VIF: original vs " + compared_to + ": " + quantize_decimal(Decimal(res)))
    return res

# SCC
def call_scc(original_image: np.ndarray, deformed_image: np.ndarray, compared_to: str):
    res = sfr.scc(original_image, deformed_image)
    print("SCC: original vs " + compared_to + ": " + quantize_decimal(Decimal(res)))
    return res

# SAM
def call_sam(original_image: np.ndarray, deformed_image: np.ndarray, compared_to: str):
    res = sfr.sam(original_image, deformed_image)
    print("SAM: original vs " + compared_to + ": " + quantize_decimal(Decimal(res)))
    return res

# helper methods
def quantize_decimal(input: Decimal) -> str:
    return str(input.quantize(Decimal('0.001'), ROUND_HALF_UP))