import numpy as np
import utils.image_tools as image_tools
import utils.noise_tools as noise_tools
import metrics.metrics_comparisons as mcc
import utils.plotting_tools as pt
from classes.ResultHolder import ResultHolder
from utils import consts
import os
import utils.data_tools as dt

def main():
    # metric testing
    # image_processing()
    # model training
    model_processing()


def model_processing():
    # data = dt.kadid_data()
    metrics = ["mse", "ergas", "psnr", "ssim", "ms-ssim", "vif", "scc", "sam"]
    for metric in metrics:
        data = image_tools.get_kadid_images_with_metric_values(metric)
        # for i in range(10):
        #     image_tools.show_image(str(i), data.images[i])
        #     print(data.metric_values[i])
        print("Transforming datas to np arrays...")
        dt.compile_model(np.array(data.images), np.array(data.metric_values), metric)
    return


def image_processing():
    # original_image = image_tools.load_image('../assets/nature.jpg')
    # image_tools.show_image("original", original_image)

    path_to_images = "../assets/kadid_ref_images"

    for image_file in os.listdir(path_to_images):
        image_path = os.path.join(path_to_images, image_file)

        image_name = image_file.split('.')[0]
        image = image_tools.load_image(image_path, image_name)
        image_tools.show_image(image_name, image)

        # create result holders for plotting
        rotation_result_holder = ResultHolder("rotations", image_name)
        salt_pepper_result_holder = ResultHolder("salt&pepper", image_name)
        gaussian_result_holder = ResultHolder("gaussian", image_name)
        # poisson_result_holder = ResultHolder("poisson", image_name)
        blur_result_holder = ResultHolder("blur", image_name)
        fade_result_holder = ResultHolder("fade", image_name)
        saturation_low_result_holder = ResultHolder("saturation_low", image_name)
        saturation_high_result_holder = ResultHolder("saturation_high", image_name)
        contrast_dark_result_holder = ResultHolder("contrast_dark", image_name)
        contrast_light_result_holder = ResultHolder("contrast_light", image_name)
        zoom_result_holder = ResultHolder("zoom", image_name)
        # salt_pepper_with_rotation_result_holder = ResultHolder("salt&pepper + 180 rotation")
        # gaussian_with_rotation_result_holder = ResultHolder("gaussian + 180 rotation")
        # poisson_with_rotation_result_holder = ResultHolder("poisson + 180 rotation")

        # # original vs rotated
        # print("Comparison: original vs 180 rotated")
        # call_comparison(original_image, rotate = True)

        # original vs zoomed
        print("Comparison: zoom with different param values")
        for zoom in consts.ZOOM_VALUES:
            print(f"Zoom value: {zoom}")
            call_comparison(image, zoom_result_holder, noise_type = "zoom", zoom = zoom)

        # plotting the results
        pt.create_plots_from_object(zoom_result_holder, consts.ZOOM_VALUES, "zoom values", "zoom")

        # original vs saturated
        print("Comparison: contrast (dark) with different param values")
        for alpha in consts.CONTRAST_DARK:
            print(f"Contrast value: {alpha}")
            call_comparison(image, contrast_dark_result_holder, noise_type = "contrast", alpha = alpha)

        # plotting the results
        pt.create_plots_from_object(contrast_dark_result_holder, consts.CONTRAST_DARK, "contrast (dark) values", "contrast_dark")

        # original vs saturated
        print("Comparison: contrast (light) with different param values")
        for alpha in consts.CONTRAST_LIGHT:
            print(f"Contrast value: {alpha}")
            call_comparison(image, contrast_light_result_holder, noise_type = "contrast", alpha = alpha)

        # plotting the results
        pt.create_plots_from_object(contrast_light_result_holder, consts.CONTRAST_LIGHT, "contrast (light) values", "contrast_light")

        # original vs saturated
        print("Comparison: saturation (low) with different param values")
        for percent in consts.SATURATION_LOW:
            print(f"Saturation (low) value: {percent}")
            call_comparison(image, saturation_low_result_holder, noise_type = "saturation", saturation = percent)

        # plotting the results
        pt.create_plots_from_object(saturation_low_result_holder, consts.SATURATION_LOW, "saturation (low) values", "saturation_low")

        # original vs saturated
        print("Comparison: saturation (high) with different param values")
        for percent in consts.SATURATION_HIGH:
            print(f"Saturation (high) value: {percent}")
            call_comparison(image, saturation_high_result_holder, noise_type = "saturation", saturation = percent)

        # plotting the results
        pt.create_plots_from_object(saturation_high_result_holder, consts.SATURATION_HIGH, "saturation (high) values", "saturation_high")

        # original vs faded
        print("Comparison: fade with different param values")
        for percent in consts.FADE_VALUES:
            print(f"Fade value: {percent}")
            call_comparison(image, fade_result_holder, noise_type = "fade", fade_percent = percent)

        # plotting the results
        pt.create_plots_from_object(fade_result_holder, consts.FADE_VALUES, "fade values", "faded")
            
        # original vs blurred
        print("Comparison: blur with different param values")
        for ksize in consts.KERNEL_SIZES:
            print(f"Blur value: {ksize}")
            call_comparison(image, blur_result_holder, noise_type = "blur", blur = ksize)
        
        # plotting the results
        pt.create_plots_from_object(blur_result_holder, consts.KERNEL_SIZES, "blur values", "blurred")

        # original vs rotated with different angles
        print("Comparison: rotated with different angles")
        for angle in consts.ANGLES:
            print(f"Angle: {angle} degrees")
            call_comparison(image, rotation_result_holder, True, angle)
        
        # plotting the results
        pt.create_plots_from_object(rotation_result_holder, consts.ANGLES, "angles (in degree)", "rotation", None, None)

        # original vs salt&pepper with different pixel values to transform
        print("Comparison: salt&pepper with different param values")
        for value in consts.PIXELS_TO_TRANSFORM:
            print(f"Number of pixels to transform: {value}")
            call_comparison(image, salt_pepper_result_holder, noise_type = "salt&pepper", number_of_pixels_to_transform = value)
        
        # plotting the results
        pt.create_plots_from_object(salt_pepper_result_holder, consts.PIXELS_TO_TRANSFORM, "pixels transformed", "salt&pepper")

        # original vs gaussian with different param values
        print("Comparison: gaussian with different param values")
        for i in range(5):
            print(f"Mean: {consts.MEANS[i]}, sigma: {consts.SIGMAS[i]}")
            call_comparison(image, gaussian_result_holder, noise_type = "gaussian", mean = consts.MEANS[i], sigma = consts.SIGMAS[i])

        # plotting the results
        pt.create_plots_from_object(gaussian_result_holder, consts.MEANS, "mean", "gaussian", consts.SIGMAS, "sigma")

        # # original vs poisson with different param values
        # print("Comparison: poisson with different param values")
        # for value in consts.GAMMAS:
        #     print(f"Gamma: {value}")
        #     call_comparison(image, poisson_result_holder, noise_type = "poisson", gamma = value)

        # # plotting the results
        # pt.create_plots_from_object(poisson_result_holder, consts.GAMMAS, "gamma", "poisson")

        # # original vs salt&pepper + 180 rotation with different pixel values to transform
        # print("Comparison: salt&pepper + 180 rotation with different param values")
        # for value in consts.PIXELS_TO_TRANSFORM:
        #     print(f"Number of pixels to transform: {value}")
        #     call_comparison(original_image, salt_pepper_with_rotation_result_holder, rotate = True, noise_type = "salt&pepper", number_of_pixels_to_transform = value)

        # # plotting the results
        # pt.create_plots_from_object(salt_pepper_with_rotation_result_holder, consts.PIXELS_TO_TRANSFORM, "pixels transformed", "salt&pepper with rotation")

        # # original vs gaussian + 180 rotation with different param values
        # print("Comparison: gaussian with different param values")
        # for i in range(8):
        #     print(f"Mean: {consts.MEANS[i]}, sigma: {consts.SIGMAS[i]}")
        #     call_comparison(original_image, gaussian_with_rotation_result_holder, rotate = True, noise_type = "gaussian", mean = consts.MEANS[i], sigma = consts.SIGMAS[i])

        # # plotting the results
        # pt.create_plots_from_object(gaussian_with_rotation_result_holder, consts.MEANS, "means", "gaussian with rotation", consts.SIGMAS, "sigma")

        # # original vs poisson + 180 rotation with different param values
        # print("Comparison: poisson with different param values")
        # for value in consts.GAMMAS:
        #     print(f"Gamma: {value}")
        #     call_comparison(original_image, poisson_with_rotation_result_holder, rotate = True, noise_type = "poisson", gamma = value)

        # # plotting the results
        # pt.create_plots_from_object(poisson_with_rotation_result_holder, consts.GAMMAS, "gamma", "poisson with rotation")


def call_comparison(image: np.ndarray, result_holder: ResultHolder, rotate: bool = False, angle: int = 180, noise_type: str = "", number_of_pixels_to_transform: int = 15000, mean: float = 0.5, sigma: int = 100,
                    gamma: float = 0.5, blur: list = (5,5), fade_percent: float = 0.2, saturation: float = 0.2, alpha: float = 0.5, zoom: float = 1.5) -> None:
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


if __name__ == "__main__":
    main()
    