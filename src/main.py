import numpy as np
import utils.image_tools as image_tools
import utils.noise_tools as noise_tools
import metrics.metrics_comparisons as mcc
import utils.plotting_tools as pt
from classes.ResultHolder import ResultHolder
from utils import consts
import matplotlib.pyplot as plt

def main():
    original_image = image_tools.load_image('../assets/nature.jpg')
    image_tools.show_image("original", original_image)

    # create result holders for plotting
    rotation_result_holder = ResultHolder("rotations")
    salt_pepper_result_holder = ResultHolder("salt&pepper")
    gaussian_result_holder = ResultHolder("gaussian")
    poisson_result_holder = ResultHolder("poisson")
    blur_result_holder = ResultHolder("blur")
    salt_pepper_with_rotation_result_holder = ResultHolder("salt&pepper + 180 rotation")
    gaussian_with_rotation_result_holder = ResultHolder("gaussian + 180 rotation")
    poisson_with_rotation_result_holder = ResultHolder("poisson + 180 rotation")

    # # original vs rotated
    # print("Comparison: original vs 180 rotated")
    # call_comparison(original_image, rotate = True)

    # original vs blurred
    print("Comparison: blur with different param values")
    for ksize in consts.KERNEL_SIZES:
        print(f"Blur value: {ksize}")
        call_comparison(original_image, blur_result_holder, noise_type = "blur", blur = ksize)
    
    # plotting the results
    pt.create_plots_from_object(blur_result_holder, consts.KERNEL_SIZES, "blur values", "blurred")

    # original vs rotated with different angles
    print("Comparison: rotated with different angles")
    for angle in consts.ANGLES:
        print(f"Angle: {angle} degrees")
        call_comparison(original_image, rotation_result_holder, True, angle)
    
    # plotting the results
    pt.create_plots_from_object(rotation_result_holder, consts.ANGLES, "angles (in degree)", "rotation", None, None)

    # original vs salt&pepper with different pixel values to transform
    print("Comparison: salt&pepper with different param values")
    for value in consts.PIXELS_TO_TRANSFORM:
        print(f"Number of pixels to transform: {value}")
        call_comparison(original_image, salt_pepper_result_holder, noise_type = "salt&pepper", number_of_pixels_to_transform = value)
    
    # plotting the results
    pt.create_plots_from_object(salt_pepper_result_holder, consts.PIXELS_TO_TRANSFORM, "pixels transformed", "salt&pepper")

    # original vs gaussian with different param values
    print("Comparison: gaussian with different param values")
    for i in range(8):
        print(f"Mean: {consts.MEANS[i]}, sigma: {consts.SIGMAS[i]}")
        call_comparison(original_image, gaussian_result_holder, noise_type = "gaussian", mean = consts.MEANS[i], sigma = consts.SIGMAS[i])

    # plotting the results
    pt.create_plots_from_object(gaussian_result_holder, consts.MEANS, "mean", "gaussian", consts.SIGMAS, "sigma")

    # original vs poisson with different param values
    print("Comparison: poisson with different param values")
    for value in consts.GAMMAS:
        print(f"Gamma: {value}")
        call_comparison(original_image, poisson_result_holder, noise_type = "poisson", gamma = value)

    # plotting the results
    pt.create_plots_from_object(poisson_result_holder, consts.GAMMAS, "gamma", "poisson")

    # original vs salt&pepper + 180 rotation with different pixel values to transform
    print("Comparison: salt&pepper + 180 rotation with different param values")
    for value in consts.PIXELS_TO_TRANSFORM:
        print(f"Number of pixels to transform: {value}")
        call_comparison(original_image, salt_pepper_with_rotation_result_holder, rotate = True, noise_type = "salt&pepper", number_of_pixels_to_transform = value)

    # plotting the results
    pt.create_plots_from_object(salt_pepper_with_rotation_result_holder, consts.PIXELS_TO_TRANSFORM, "pixels transformed", "salt&pepper with rotation")

    # original vs gaussian + 180 rotation with different param values
    print("Comparison: gaussian with different param values")
    for i in range(8):
        print(f"Mean: {consts.MEANS[i]}, sigma: {consts.SIGMAS[i]}")
        call_comparison(original_image, gaussian_with_rotation_result_holder, rotate = True, noise_type = "gaussian", mean = consts.MEANS[i], sigma = consts.SIGMAS[i])

    # plotting the results
    pt.create_plots_from_object(gaussian_with_rotation_result_holder, consts.MEANS, "means", "gaussian with rotation", consts.SIGMAS, "sigma")

    # original vs poisson + 180 rotation with different param values
    print("Comparison: poisson with different param values")
    for value in consts.GAMMAS:
        print(f"Gamma: {value}")
        call_comparison(original_image, poisson_with_rotation_result_holder, rotate = True, noise_type = "poisson", gamma = value)

    # plotting the results
    pt.create_plots_from_object(poisson_with_rotation_result_holder, consts.GAMMAS, "gamma", "poisson with rotation")



def call_comparison(image: np.ndarray, result_holder: ResultHolder, rotate: bool = False, angle: int = 180, noise_type: str = "", number_of_pixels_to_transform: int = 15000, mean: float = 0.5, sigma: int = 100,
                    gamma: float = 0.5, blur: list = (5,5)) -> None:
    metrics = ["mse", "ergas", "psnr", "ssim", "ms-ssim", "vif", "scc", "sam"]
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

    for metric in metrics:
        result = mcc.call_concrete_comparison(image, metric, rotate, angle, noise_type, number_of_pixels_to_transform, mean, sigma, gamma, blur)
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


if __name__ == "__main__":
    main()
    