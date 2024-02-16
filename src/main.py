import numpy as np
import utils.image_tools as image_tools
import utils.noise_tools as noise_tools
import metrics.metrics_comparisons as mcc
import utils.plotting_tools as pt
from utils.results import mse_rotation_results, ms_ssim_rotation_results, ergas_rotation_results, psnr_rotation_results, sam_rotation_results, scc_rotation_results, ssim_rotation_results, vif_rotation_results

def main():
    original_image = image_tools.load_image('../assets/nature.jpg')
    image_tools.show_image("original", original_image)

    # noise param variables
    pixels_to_transform = [5000, 15000, 25000, 50000, 75000]
    means = [10, 50, 100, 200, 400]
    sigmas = [50, 100, 200, 400, 800]
    gammas= [0.25, 1, 1.5, 2, 2.5]
    angles = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    test = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    


    # # original vs rotated
    # print("Comparison: original vs 180 rotated")
    # call_comparison(original_image, rotate = True)

    # original vs rotated with different angles
    print("Comparison: rotated with different angles")
    for angle in angles:
        print(f"Angle: {angle} degrees")
        call_comparison(original_image, True, angle)

    pt.create_plot_for_metric(angles, "angles (in degree)", mse_rotation_results, "mse values", "mse vs rotations")
    pt.create_plot_for_metric(angles, "angles (in degree)", ergas_rotation_results, "ergas values", "ergas vs rotations")
    pt.create_plot_for_metric(angles, "angles (in degree)", psnr_rotation_results, "psnr values", "psn vs rotations")
    pt.create_plot_for_metric(angles, "angles (in degree)", ssim_rotation_results, "ssim values", "ssim vs rotations")
    pt.create_plot_for_metric(angles, "angles (in degree)", ms_ssim_rotation_results, "ms-ssim values", "ms-ssim vs rotations")
    pt.create_plot_for_metric(angles, "angles (in degree)", vif_rotation_results, "vif values", "vif vs rotations")
    pt.create_plot_for_metric(angles, "angles (in degree)", scc_rotation_results, "scc values", "scc vs rotations")
    pt.create_plot_for_metric(angles, "angles (in degree)", sam_rotation_results, "sam values", "sam vs rotations")

    # # original vs salt&pepper with different pixel values to transform
    # print("Comparison: salt&pepper with different param values")
    # for value in pixels_to_transform:
    #     print(f"Number of pixels to transform: {value}")
    #     call_comparison(original_image, noise_type = "salt&pepper", number_of_pixels_to_transform = value)

    # # original vs gaussian with different param values
    # print("Comparison: gaussian with different param values")
    # for i in range(5):
    #     print(f"Mean: {means[i]}, sigma: {sigmas[i]}")
    #     call_comparison(original_image, noise_type = "gaussian", mean = means[i], sigma = sigmas[i])

    # # original vs poisson with different param values
    # print("Comparison: poisson with different param values")
    # for i in range(5):
    #     print(f"Gamma: {gammas[i]}")
    #     call_comparison(original_image, noise_type = "poisson", gamma = gammas[i])

    # # original vs salt&pepper + 180 rotation with different pixel values to transform
    # print("Comparison: salt&pepper + 180 rotation with different param values")
    # for value in pixels_to_transform:
    #     print(f"Number of pixels to transform: {value}")
    #     call_comparison(original_image, rotate = True, noise_type = "salt&pepper", number_of_pixels_to_transform = value)

    # # original vs gaussian + 180 rotation with different param values
    # print("Comparison: gaussian with different param values")
    # for i in range(5):
    #     print(f"Mean: {means[i]}, sigma: {sigmas[i]}")
    #     call_comparison(original_image, rotate = True, noise_type = "gaussian", mean = means[i], sigma = sigmas[i])

    # # original vs poisson + 180 rotation with different param values
    # print("Comparison: poisson with different param values")
    # for i in range(5):
    #     print(f"Gamma: {gammas[i]}")
    #     call_comparison(original_image, rotate = True, noise_type = "poisson", gamma = gammas[i])



def call_comparison(image: np.ndarray, rotate: bool = False, angle: int = 180, noise_type: str = "", number_of_pixels_to_transform: int = 15000, mean: float = 0.5, sigma: int = 100,
                    gamma: float = 0.5) -> None:
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
        image_tools.show_image("poisson", noise_tools.poisson(image,gamma))
    elif noise_type == "poisson" and rotate:
        print(f"poisson noise comparison (gamma: {gamma})")
        image_tools.show_image("poisson", image_tools.generate_180_rotated_with_noise(image, "poisson", gamma = gamma))

    for metric in metrics:
        result = mcc.call_concrete_comparison(image, metric, rotate,angle, noise_type, number_of_pixels_to_transform, mean, sigma, gamma)
        if metric == "mse":
            mse_rotation_results.append(result)
        elif metric == "ergas":
            ergas_rotation_results.append(result)
        elif metric == "psnr":
            psnr_rotation_results.append(result)
        elif metric == "ssim":
            ssim_rotation_results.append(result)
        elif metric == "ms-ssim":
            ms_ssim_rotation_results.append(result)
        elif metric == "vif":
            vif_rotation_results.append(result)
        elif metric == "scc":
            scc_rotation_results.append(result)
        elif metric == "sam":
            sam_rotation_results.append(result)


if __name__ == "__main__":
    main()
    