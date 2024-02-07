import numpy as np
import utils.image_tools as image_tools
import metrics.metrics_caller as mc

def call_concrete_comparison(original_image: np.ndarray, metric: str, rotate: bool, noise_type: str = "salt&pepper", number_of_pixels_to_transform: int = 15000, 
                   mean: float = 0.5, sigma: float = 100, gamma: float = 0.5) -> None:
    if noise_type and not rotate:
        deformed = image_tools.create_concrete_noisy_image(original_image, noise_type, number_of_pixels_to_transform, mean, sigma, gamma)
        compared_to = f"{noise_type} noised"
    elif noise_type and rotate:
        deformed = image_tools.generate_180_rotated_with_noise(original_image, noise_type, number_of_pixels_to_transform, mean, sigma, gamma)
        compared_to = f"{noise_type} noised + 180 rotated"
    else:
        deformed = image_tools.generate_180_rotated(original_image)
        compared_to = "180 roted"

    call_prints(original_image, deformed, metric, compared_to)

def call_prints(original_image: np.ndarray, deformed: np.ndarray, metric: str, compared_to: str) -> None:
    # write to file here ? 
    if metric == "mse":
        print(mc.call_mse(original_image, deformed, compared_to))
    elif metric == "ergas":
        print(mc.call_ergas(original_image, deformed, compared_to))
    elif metric == "psnr":
        print(mc.call_psnr(original_image, deformed, compared_to))
    elif metric == "ssim":
        print(mc.call_ssim(original_image, deformed, compared_to))
    elif metric == "ms-ssim":
        print(mc.call_msssim(original_image, deformed, compared_to))
    elif metric == "vif":
        print(mc.call_vif(original_image, deformed, compared_to))
    elif metric == "scc":
        print(mc.call_scc(original_image, deformed, compared_to))
    elif metric == "sam":
        print(mc.call_sam(original_image, deformed, compared_to))