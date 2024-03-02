import numpy as np
import utils.image_tools as image_tools
import metrics.metrics_caller as mc

def call_concrete_comparison(original_image: np.ndarray, metric: str, rotate: bool, angle: int = 180, noise_type: str = "salt&pepper", number_of_pixels_to_transform: int = 15000, 
                   mean: float = 0.5, sigma: float = 100, gamma: float = 0.5, blur: list = (5,5), fade_percent: float = 0.2, saturation: float = 0.2):
    if noise_type and not rotate:
        deformed = image_tools.create_concrete_noisy_image(original_image, noise_type, number_of_pixels_to_transform, mean, sigma, gamma, blur, fade_percent, saturation)
        compared_to = f"{noise_type} noised"
    elif noise_type and rotate:
        deformed = image_tools.generate_180_rotated_with_noise(original_image, noise_type, number_of_pixels_to_transform, mean, sigma, gamma, blur, fade_percent, saturation)
        compared_to = f"{noise_type} noised + {angle} rotated"
    else:
        deformed = image_tools.create_rotated_image(original_image, angle)
        compared_to = f"{angle} roted"

    return call_prints(original_image, deformed, metric, compared_to)

def call_prints(original_image: np.ndarray, deformed: np.ndarray, metric: str, compared_to: str):
    # write to file here ? 
    if metric == "mse":
        return(mc.call_mse(original_image, deformed, compared_to))
    elif metric == "ergas":
        return(mc.call_ergas(original_image, deformed, compared_to))
    elif metric == "psnr":
        return(mc.call_psnr(original_image, deformed, compared_to))
    elif metric == "ssim":
        return(mc.call_ssim(original_image, deformed, compared_to))
    elif metric == "ms-ssim":
        return(mc.call_msssim(original_image, deformed, compared_to))
    elif metric == "vif":
        return(mc.call_vif(original_image, deformed, compared_to))
    elif metric == "scc":
        return(mc.call_scc(original_image, deformed, compared_to))
    elif metric == "sam":
        return(mc.call_sam(original_image, deformed, compared_to))