import numpy as np
from classes.ResultHolder import ResultHolder
import utils.image_tools as image_tools
import metrics.metrics_caller as mc
import utils.noise_tools as noise_tools
import metrics.metrics_comparisons as mcc

def call_comparison(image: np.ndarray, result_holder: ResultHolder, rotate: bool = False, angle: int = 180, noise_type: str = "", number_of_pixels_to_transform: int = 15000, mean: float = 0.5, sigma: int = 100,
                    gamma: float = 0.5, blur: list = (5,5), fade_percent: float = 0.2, saturation: float = 0.2, alpha: float = 0.5, zoom: float = 1.5) -> None:
    '''
    Function to display the currently processed image.
    This contains all the noise parameters, to handle them in one function, this could be separated in the future... 
    '''
    # define metrics for the iteration
    metrics = ["mse", "ergas", "psnr", "ssim", "ms-ssim", "vif", "scc", "sam"]

    # display proper image
    if noise_type == "":
        print("rotated comparison")
        # image_tools.show_image("rotated", image_tools.create_rotated_image(image, angle))
    elif noise_type == "salt&pepper" and not rotate:
        print(f"salt&pepper noise comparison ({number_of_pixels_to_transform} - pixel transformed)")
        image_tools.show_image("salt&pepper", noise_tools.salt_and_pepper(image, number_of_pixels_to_transform))
    elif noise_type == "salt&pepper" and rotate:
        print(f"salt&pepper noise + 180 rotation comparison ({number_of_pixels_to_transform} - pixel transformed)")
        image_tools.show_image("salt&pepper", image_tools.generate_180_rotated_with_noise(image, "salt&pepper", number_of_pixels_to_transform))
    elif noise_type == "gaussian" and not rotate:
        print(f"gaussian noise comparison (mean: {mean} and sigma: {sigma})")
        image_tools.show_image("gaussian", noise_tools.gaussian(image, mean, sigma))
    elif noise_type == "gaussian" and rotate:
        print(f"gaussian noise comparison (mean: {mean} and sigma: {sigma})")
        image_tools.show_image("gaussian", image_tools.generate_180_rotated_with_noise(image, "gaussian", mean = mean, sigma = sigma))
    elif noise_type == "poisson" and not rotate:
        print(f"poisson noise comparison (gamma: {gamma})")
        image_tools.show_image("poisson", noise_tools.poisson(image, gamma))
    elif noise_type == "poisson" and rotate:
        print(f"poisson noise comparison (gamma: {gamma})")
        image_tools.show_image("poisson", image_tools.generate_180_rotated_with_noise(image, "poisson", gamma = gamma))
    elif noise_type == "blur" and not rotate:
        print(f"blur comparison (values: {blur})")
        image_tools.show_image("blur", noise_tools.blur(image, blur))
    elif noise_type == "blur" and rotate:
        print(f"blur comparison with rotation (values: {blur}")
        image_tools.show_image("blur", image_tools.generate_180_rotated_with_noise(image, "blur", blur = blur))
    elif noise_type == "fade" and not rotate:
        print(f"fade comparison (fade percent: {fade_percent})")
        image_tools.show_image("fade", noise_tools.fade(image, fade_percent))
    elif noise_type == "fade" and rotate:
        print(f"fade comparison with rotation (fade percent: {fade_percent}")
        image_tools.show_image("fade", image_tools.generate_180_rotated_with_noise(image, "fade", fade_percent = fade_percent))
    elif noise_type == "saturation" and not rotate:
        print(f"saturation comparison (saturation percent: {saturation})")
        image_tools.show_image("saturation", noise_tools.saturation(image, saturation))
    elif noise_type == "saturation" and rotate:
        print(f"saturation comparison with rotation (saturation percent: {saturation}")
        image_tools.show_image("saturation", image_tools.generate_180_rotated_with_noise(image, "saturation", saturation = saturation))
    elif noise_type == "contrast" and not rotate:
        print(f"contrast comparison (alpha: {alpha})")
        image_tools.show_image("contrast", noise_tools.contrast(image, alpha))
    elif noise_type == "contrast" and rotate:
        print(f"contrast comparison with rotation (alpha: {alpha}")
        image_tools.show_image("contrast", image_tools.generate_180_rotated_with_noise(image, "contrast", alpha = alpha))
    elif noise_type == "zoom" and not rotate:
        print(f"zoom comparison (zoom value: {zoom})")
        image_tools.show_image("zoom", noise_tools.zoom(image, zoom))
    elif noise_type == "contrast" and rotate:
        print(f"zoom comparison with rotation (zoom value: {zoom}")
        image_tools.show_image("zoom", image_tools.generate_180_rotated_with_noise(image, "zoom", zoom = zoom))

    # iterate trough metrics, create the concrete noised image, calculate metric values, and append the results to the proper array of the resultholder
    # this also can be simplified in the future..
    for metric in metrics:
        result = mcc.call_concrete_comparison(image, metric, rotate, angle, noise_type, number_of_pixels_to_transform, mean, sigma, gamma, blur, fade_percent, saturation, alpha, zoom)
        if metric == "mse":
            result_holder.mse_results.append(result)
        elif metric == "ergas":
            result_holder.ergas_results.append(result)
        elif metric == "psnr":
            result_holder.psnr_results.append(result)
        elif metric == "ssim":
            result_holder.ssim_results.append(result)
        elif metric == "ms-ssim":
            result_holder.ms_ssim_results.append(result)
        elif metric == "vif":
            result_holder.vif_results.append(result)
        elif metric == "scc":
            result_holder.scc_results.append(result)
        elif metric == "sam":
            result_holder.sam_results.append(result)

def call_concrete_comparison(original_image: np.ndarray, metric: str, rotate: bool, angle: int = 180, noise_type: str = "salt&pepper", number_of_pixels_to_transform: int = 15000, 
                   mean: float = 0.5, sigma: float = 100, gamma: float = 0.5, blur: list = (5,5), fade_percent: float = 0.2, saturation: float = 0.2, alpha: float = 0.5, 
                   zoom: float = 1.5):
    '''
    Create the concrete deformed image based on the incooming parameters.
    Then call the function that calls directly the metric functions.
    '''
    # determine if the method should create only a noised image or rotated image, or a combo of a noise and rotation 
    if noise_type and not rotate:
        deformed = image_tools.create_concrete_noisy_image(original_image, noise_type, number_of_pixels_to_transform, mean, sigma, gamma, blur, fade_percent, saturation, alpha, zoom)
        compared_to = f"{noise_type} noised"
    elif noise_type and rotate:
        deformed = image_tools.generate_180_rotated_with_noise(original_image, noise_type, number_of_pixels_to_transform, mean, sigma, gamma, blur, fade_percent, saturation, alpha, zoom)
        compared_to = f"{noise_type} noised + {angle} rotated"
    else:
        deformed = image_tools.create_rotated_image(original_image, angle)
        compared_to = f"{angle} roted"

    # call print method
    return call_prints(original_image, deformed, metric, compared_to)

def call_prints(original_image: np.ndarray, deformed: np.ndarray, metric: str, compared_to: str):
    '''
    Function to call the concrete metric calculation values.
    '''
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