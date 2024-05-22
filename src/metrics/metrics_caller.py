# with the defined methods in this file, metrics can be called in a simple way
import math
import sewar.full_ref as sfr
import numpy as np
from decimal import Decimal, ROUND_HALF_UP

# MSE
def call_mse(original_image: np.ndarray, deformed_image: np.ndarray, compared_to: str):
    '''
    Calculate mse value of based on the incoming ref and deformed image.
    '''
    res = sfr.mse(original_image, deformed_image)
    # print("MSE: original vs " + compared_to + ": " + quantize_decimal(Decimal(res)))
    return res

# ERGAS
def call_ergas(original_image: np.ndarray, deformed_image: np.ndarray, compared_to: str):
    '''
    Calculate ergas value of based on the incoming ref and deformed image.
    '''
    res = sfr.ergas(original_image, deformed_image)
    # print("ERGAS: original vs " + compared_to + ": " + quantize_decimal(Decimal(res)))
    return res

# PSNR
def call_psnr(original_image: np.ndarray, deformed_image: np.ndarray, compared_to: str):
    '''
    Calculate psnr value of based on the incoming ref and deformed image.
    '''
    res = sfr.psnr(original_image, deformed_image)
    if math.isinf(res): return 0.0
    # print("PSNR: original vs " + compared_to + ": " + quantize_decimal(Decimal(res)))
    return res

# SSIM
def call_ssim(original_image: np.ndarray, deformed_image: np.ndarray, compared_to: str):
    '''
    Calculate ssim value of based on the incoming ref and deformed image.
    '''
    res = sfr.ssim(original_image, deformed_image)[0]
    # print("SSIM: original vs " + compared_to + ": " + quantize_decimal(Decimal(res)))
    return res

# MS-SSIM
def call_msssim(original_image: np.ndarray, deformed_image: np.ndarray, compared_to: str):
    '''
    Calculate ms-ssim value of based on the incoming ref and deformed image.
    '''
    res = sfr.msssim(original_image, deformed_image).real
    # print("MS-SSIM: original vs " + compared_to + ": " + str(res))
    return res

# VIF
def call_vif(original_image: np.ndarray, deformed_image: np.ndarray, compared_to: str):
    '''
    Calculate vif value of based on the incoming ref and deformed image.
    '''
    res = sfr.vifp(original_image, deformed_image)
    # print("VIF: original vs " + compared_to + ": " + quantize_decimal(Decimal(res)))
    return res

# SCC
def call_scc(original_image: np.ndarray, deformed_image: np.ndarray, compared_to: str):
    '''
    Calculate scc value of based on the incoming ref and deformed image.
    '''
    res = sfr.scc(original_image, deformed_image)
    # print("SCC: original vs " + compared_to + ": " + quantize_decimal(Decimal(res)))
    return res

# SAM
def call_sam(original_image: np.ndarray, deformed_image: np.ndarray, compared_to: str):
    '''
    Calculate sam value of based on the incoming ref and deformed image.
    '''
    res = sfr.sam(original_image, deformed_image)
    # print("SAM: original vs " + compared_to + ": " + quantize_decimal(Decimal(res)))
    return res

# helper methods
def quantize_decimal(input: Decimal) -> str:
    '''
    Function to round metric values to 3 decimal places.
    '''
    return str(input.quantize(Decimal('0.001'), ROUND_HALF_UP))