import numpy as np
import utils.image_tools as image_tools
import utils.noise_tools as noise_tools
import metrics.metrics_comparisons as mcc
import metrics.metrics_caller as mc

def main():
    original_image = image_tools.load_image('../assets/nature.jpg')
    image_tools.show_image("original", original_image)

    # noise param variables
    pixels_to_transform = [5000, 15000, 25000, 50000, 75000]
    means = [10, 50, 100, 200, 400]
    sigmas = [50, 100, 200, 400, 800]
    gammas= [0.25, 1, 1.5, 2, 2.5]

    # original vs rotated
    print("Comparison: original vs 180 rotated")
    call_comparison(original_image, rotate = True)

    # # original vs salt&pepper with different pixel values to transform
    # print("Comparison: salt&pepper with different param values")
    # for value in pixels_to_transform:
    #     print(f"Number of pixels to transform: {value}")
    #     call_comparison(original_image, noise_type = "salt&pepper", number_of_pixels_to_transform = value)

    # original vs gaussian with different param values
    print("Comparison: gaussian with different param values")
    for i in range(5):
        print(f"Mean: {means[i]}, sigma: {sigmas[i]}")
        call_comparison(original_image, noise_type = "gaussian", mean = means[i], sigma = sigmas[i])

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

    # original vs gaussian + 180 rotation with different param values
    print("Comparison: gaussian with different param values")
    for i in range(5):
        print(f"Mean: {means[i]}, sigma: {sigmas[i]}")
        call_comparison(original_image, rotate = True, noise_type = "gaussian", mean = means[i], sigma = sigmas[i])

    # # original vs poisson + 180 rotation with different param values
    # print("Comparison: poisson with different param values")
    # for i in range(5):
    #     print(f"Gamma: {gammas[i]}")
    #     call_comparison(original_image, rotate = True, noise_type = "poisson", gamma = gammas[i])



def call_comparison(image: np.ndarray, rotate: bool = False, noise_type: str = "", number_of_pixels_to_transform: int = 15000, mean: float = 0.5, sigma: int = 100,
                    gamma: float = 0.5) -> None:
    metrics = ["mse", "ergas", "psnr", "ssim", "ms-ssim", "vif", "scc", "sam"]
    if noise_type == "":
        print("rotated comparison")
        image_tools.show_image("rotated", image_tools.generate_180_rotated(image))
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
        mcc.call_concrete_comparison(image, metric, rotate, noise_type, number_of_pixels_to_transform, mean, sigma, gamma)
    print("\n")


if __name__ == "__main__":
    main()
    