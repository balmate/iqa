import numpy as np
import utils.image_tools as image_tools
import utils.noise_tools as noise_tools
import metrics.metrics_comparisons as mcc
import utils.plotting_tools as pt
from classes.ResultHolder import ResultHolder
from utils import consts
import metrics.metrics_comparisons as mc

def main():
    original_image = image_tools.load_image('../assets/nature.jpg')
    image_tools.show_image("original", original_image)
    metrics = ["mse", "ergas", "psnr", "ssim", "ms-ssim", "vif", "scc", "sam"]

    # # create result holders for plotting
    # rotation_result_holder = ResultHolder("rotations")
    # salt_pepper_result_holder = ResultHolder("salt&pepper")
    # gaussian_result_holder = ResultHolder("gaussian")
    # poisson_result_holder = ResultHolder("poisson")
    # blur_result_holder = ResultHolder("blur")
    # fade_result_holder = ResultHolder("fade")
    # saturation_result_holder = ResultHolder("saturation")
    # contrast_result_holder = ResultHolder("contrast")
    # zoom_result_holder = ResultHolder("zoom") 
    # salt_pepper_with_rotation_result_holder = ResultHolder("salt&pepper + 180 rotation")
    # gaussian_with_rotation_result_holder = ResultHolder("gaussian + 180 rotation")
    # poisson_with_rotation_result_holder = ResultHolder("poisson + 180 rotation")
    
    blur_with_salt_and_pepper_result_holder = ResultHolder("(7,7) blur with salt&pepper")
    blur = (5,5)
    # original vs zoomed
    print("Comparison: (7,7) blur with different salt&pepper values")
    for value in consts.PIXELS_TO_TRANSFORM:
        print(f"modified pixels: {value}")
        deformed = image_tools.blur_with_salt_and_pepper(original_image, blur, value, f"transofmed images/combos/blur_5_5_salt&pepper_{value}.jpg")
        image_tools.show_image("(7,7) blur with salt and pepper", deformed)
        for metric in metrics:
            result = mc.call_prints(original_image, deformed, metric, f"{blur} blur with salt&pepper (transformed pixels: {value})")
            if metric == "mse":
                blur_with_salt_and_pepper_result_holder.mse_results.append(result)
            elif metric == "ergas":
                blur_with_salt_and_pepper_result_holder.ergas_results.append(result)
            elif metric == "psnr":
                blur_with_salt_and_pepper_result_holder.psnr_results.append(result)
            elif metric == "ssim":
                blur_with_salt_and_pepper_result_holder.ssim_results.append(result)
            elif metric == "ms-ssim":
                blur_with_salt_and_pepper_result_holder.ms_ssim_results.append(result)
            elif metric == "vif":
                blur_with_salt_and_pepper_result_holder.vif_results.append(result)
            elif metric == "scc":
                blur_with_salt_and_pepper_result_holder.scc_results.append(result)
            elif metric == "sam":
                blur_with_salt_and_pepper_result_holder.sam_results.append(result)

    # plotting the results
    pt.create_plots_from_object(blur_with_salt_and_pepper_result_holder, consts.PIXELS_TO_TRANSFORM, "pixels transformed", "(5,5) blur with salt&pepper")

    rotate_with_salt_and_pepper_result_holder = ResultHolder("rotated with salt&pepper (10000)")
    salt_and_pepper = 5000
    # original vs zoomed
    print("Comparison: rotated with different salt&pepper values")
    for angle in consts.ANGLES:
        print(f"rotation value: {angle}")
        deformed = image_tools.rotated_with_salt_and_pepper(original_image, angle, salt_and_pepper,
                                                             f"transformed_images/combos/rotate_{angle}_salt&pepper_{salt_and_pepper}.jpg")
        image_tools.show_image("rotate with salt and pepper (10000)", deformed)
        for metric in metrics:
            result = mc.call_prints(original_image, deformed, metric, f"({angle} rotation with salt&pepper {salt_and_pepper})")
            if metric == "mse":
                rotate_with_salt_and_pepper_result_holder.mse_results.append(result)
            elif metric == "ergas":
                rotate_with_salt_and_pepper_result_holder.ergas_results.append(result)
            elif metric == "psnr":
                rotate_with_salt_and_pepper_result_holder.psnr_results.append(result)
            elif metric == "ssim":
                rotate_with_salt_and_pepper_result_holder.ssim_results.append(result)
            elif metric == "ms-ssim":
                rotate_with_salt_and_pepper_result_holder.ms_ssim_results.append(result)
            elif metric == "vif":
                rotate_with_salt_and_pepper_result_holder.vif_results.append(result)
            elif metric == "scc":
                rotate_with_salt_and_pepper_result_holder.scc_results.append(result)
            elif metric == "sam":
                rotate_with_salt_and_pepper_result_holder.sam_results.append(result)

    # plotting the results
    pt.create_plots_from_object(rotate_with_salt_and_pepper_result_holder, consts.ANGLES, "angle", f"rotate with salt&pepper ({salt_and_pepper})")

    rotate_with_blur_result_holder = ResultHolder("rotated with blur (5,5)")
    blur2 = (5,5)
    # original vs zoomed
    print("Comparison: rotated with different blur values")
    for angle in consts.ANGLES:
        print(f"rotation value: {angle}")
        deformed = image_tools.rotated_with_blur(original_image, angle, blur2, f"transofmed_images/combos/rotated_{angle}_blur_5_5.jpg")
        image_tools.show_image("rotate with blur (5,5))", deformed)
        for metric in metrics:
            result = mc.call_prints(original_image, deformed, metric, f"({angle} rotation with blur ({blur2}))")
            if metric == "mse":
                rotate_with_blur_result_holder.mse_results.append(result)
            elif metric == "ergas":
                rotate_with_blur_result_holder.ergas_results.append(result)
            elif metric == "psnr":
                rotate_with_blur_result_holder.psnr_results.append(result)
            elif metric == "ssim":
                rotate_with_blur_result_holder.ssim_results.append(result)
            elif metric == "ms-ssim":
                rotate_with_blur_result_holder.ms_ssim_results.append(result)
            elif metric == "vif":
                rotate_with_blur_result_holder.vif_results.append(result)
            elif metric == "scc":
                rotate_with_blur_result_holder.scc_results.append(result)
            elif metric == "sam":
                rotate_with_blur_result_holder.sam_results.append(result)

    # plotting the results
    pt.create_plots_from_object(rotate_with_blur_result_holder, consts.ANGLES, "angle", "rotate with blur (5,5)")

    rotate_with_contrast_result_holder = ResultHolder("rotated with contrast (0.2)")
    contrast = 0.2
    # original vs zoomed
    print("Comparison: rotated with different contrast values")
    for angle in consts.ANGLES:
        print(f"rotation value: {angle}")
        deformed = image_tools.rotated_with_contrast(original_image, angle, contrast, f"transformed_images/combos/rotated_{angle}_contrast_0_2.jpg")
        image_tools.show_image("rotate with contrast (0.2))", deformed)
        for metric in metrics:
            result = mc.call_prints(original_image, deformed, metric, f"({angle} rotation with contrast ({contrast}))")
            if metric == "mse":
                rotate_with_contrast_result_holder.mse_results.append(result)
            elif metric == "ergas":
                rotate_with_contrast_result_holder.ergas_results.append(result)
            elif metric == "psnr":
                rotate_with_contrast_result_holder.psnr_results.append(result)
            elif metric == "ssim":
                rotate_with_contrast_result_holder.ssim_results.append(result)
            elif metric == "ms-ssim":
                rotate_with_contrast_result_holder.ms_ssim_results.append(result)
            elif metric == "vif":
                rotate_with_contrast_result_holder.vif_results.append(result)
            elif metric == "scc":
                rotate_with_contrast_result_holder.scc_results.append(result)
            elif metric == "sam":
                rotate_with_contrast_result_holder.sam_results.append(result)

    # plotting the results
    pt.create_plots_from_object(rotate_with_contrast_result_holder, consts.ANGLES, "angle", "rotate with contrast (0_2)")

    saturation_with_salt_and_pepper = ResultHolder("saturation with salt&pepper (5000)")
    salt_and_pepper2 = 5000
    # original vs zoomed
    print("Comparison: rotated with different contrast values")
    for percent in consts.SATURATION_VALUES:
        print(f"saturation value: {percent}")
        deformed = image_tools.saturation_with_salt_and_pepper(original_image, percent, salt_and_pepper2, f"transformed_images/combos/salt&pepper_{salt_and_pepper2}_saturation_{str(percent).replace('.','_')}.jpg")
        image_tools.show_image("satur with salt&pepper(5000))", deformed)
        for metric in metrics:
            result = mc.call_prints(original_image, deformed, metric, f"({percent} saturated with salt&pepper(5000))")
            if metric == "mse":
                saturation_with_salt_and_pepper.mse_results.append(result)
            elif metric == "ergas":
                saturation_with_salt_and_pepper.ergas_results.append(result)
            elif metric == "psnr":
                saturation_with_salt_and_pepper.psnr_results.append(result)
            elif metric == "ssim":
                saturation_with_salt_and_pepper.ssim_results.append(result)
            elif metric == "ms-ssim":
                saturation_with_salt_and_pepper.ms_ssim_results.append(result)
            elif metric == "vif":
                saturation_with_salt_and_pepper.vif_results.append(result)
            elif metric == "scc":
                saturation_with_salt_and_pepper.scc_results.append(result)
            elif metric == "sam":
                saturation_with_salt_and_pepper.sam_results.append(result)

    # plotting the results
    pt.create_plots_from_object(saturation_with_salt_and_pepper, consts.SATURATION_VALUES, "percent", "saturation with salt&pepper (70000)")

    fade_with_contrast_result_holder = ResultHolder("fade with contrast(0.2)")
    contrast_value = 2.5
    # original vs zoomed
    print("Comparison: rotated with different contrast values")
    for percent in consts.FADE_VALUES:
        print(f"fade value: {percent}")
        deformed = image_tools.fade_with_contrast(original_image, percent, contrast_value, f"transformed_images/combos/fade_{str(percent).replace('.', '_')}_contrast")
        image_tools.show_image(f"fade with contrast({contrast_value}))", deformed)
        for metric in metrics:
            result = mc.call_prints(original_image, deformed, metric, f"({percent} fade with contrast ({contrast_value}))")
            if metric == "mse":
                fade_with_contrast_result_holder.mse_results.append(result)
            elif metric == "ergas":
                fade_with_contrast_result_holder.ergas_results.append(result)
            elif metric == "psnr":
                fade_with_contrast_result_holder.psnr_results.append(result)
            elif metric == "ssim":
                fade_with_contrast_result_holder.ssim_results.append(result)
            elif metric == "ms-ssim":
                fade_with_contrast_result_holder.ms_ssim_results.append(result)
            elif metric == "vif":
                fade_with_contrast_result_holder.vif_results.append(result)
            elif metric == "scc":
                fade_with_contrast_result_holder.scc_results.append(result)
            elif metric == "sam":
                fade_with_contrast_result_holder.sam_results.append(result)

    # plotting the results
    pt.create_plots_from_object(fade_with_contrast_result_holder, consts.FADE_VALUES, "percent", "fade with contrast (2_5)")


    # # original vs rotated
    # print("Comparison: original vs 180 rotated")
    # call_comparison(original_image, rotate = True)
    
    # # original vs zoomed
    # print("Comparison: zoom with different param values")
    # for zoom in consts.ZOOM_VALUES:
    #     print(f"Zoom value: {zoom}")
    #     call_comparison(original_image, zoom_result_holder, noise_type = "zoom", zoom = zoom)

    # # plotting the results
    # pt.create_plots_from_object(zoom_result_holder, consts.ZOOM_VALUES, "zoom values", "zoom")

    # # original vs saturated
    # print("Comparison: contrast with different param values")
    # for alpha in consts.ALPHAS:
    #     print(f"Contrast value: {alpha}")
    #     call_comparison(original_image, contrast_result_holder, noise_type = "contrast", alpha = alpha)

    # # plotting the results
    # pt.create_plots_from_object(contrast_result_holder, consts.ALPHAS, "contrast values", "contrast")

    # # original vs saturated
    # print("Comparison: saturation with different param values")
    # for percent in consts.SATURATION_VALUES:
    #     print(f"Saturation value: {percent}")
    #     call_comparison(original_image, saturation_result_holder, noise_type = "saturation", saturation = percent)

    # # plotting the results
    # pt.create_plots_from_object(saturation_result_holder, consts.SATURATION_VALUES, "saturation values", "saturation")

    # # original vs faded
    # print("Comparison: fade with different param values")
    # for percent in consts.FADE_VALUES:
    #     print(f"Fade value: {percent}")
    #     call_comparison(original_image, fade_result_holder, noise_type = "fade", fade_percent = percent)

    # # plotting the results
    # pt.create_plots_from_object(fade_result_holder, consts.FADE_VALUES, "fade values", "faded")
        
    # # original vs blurred
    # print("Comparison: blur with different param values")
    # for ksize in consts.KERNEL_SIZES:
    #     print(f"Blur value: {ksize}")
    #     call_comparison(original_image, blur_result_holder, noise_type = "blur", blur = ksize)
    
    # # plotting the results
    # pt.create_plots_from_object(blur_result_holder, consts.KERNEL_SIZES, "blur values", "blurred")

    # # original vs rotated with different angles
    # print("Comparison: rotated with different angles")
    # for angle in consts.ANGLES:
    #     print(f"Angle: {angle} degrees")
    #     call_comparison(original_image, rotation_result_holder, True, angle)
    
    # # plotting the results
    # pt.create_plots_from_object(rotation_result_holder, consts.ANGLES, "angles (in degree)", "rotation", None, None)

    # # original vs salt&pepper with different pixel values to transform
    # print("Comparison: salt&pepper with different param values")
    # for value in consts.PIXELS_TO_TRANSFORM:
    #     print(f"Number of pixels to transform: {value}")
    #     call_comparison(original_image, salt_pepper_result_holder, noise_type = "salt&pepper", number_of_pixels_to_transform = value)
    
    # # plotting the results
    # pt.create_plots_from_object(salt_pepper_result_holder, consts.PIXELS_TO_TRANSFORM, "pixels transformed", "salt&pepper")

    # # original vs gaussian with different param values
    # print("Comparison: gaussian with different param values")
    # for i in range(8):
    #     print(f"Mean: {consts.MEANS[i]}, sigma: {consts.SIGMAS[i]}")
    #     call_comparison(original_image, gaussian_result_holder, noise_type = "gaussian", mean = consts.MEANS[i], sigma = consts.SIGMAS[i])

    # # plotting the results
    # pt.create_plots_from_object(gaussian_result_holder, consts.MEANS, "mean", "gaussian", consts.SIGMAS, "sigma")

    # # original vs poisson with different param values
    # print("Comparison: poisson with different param values")
    # for value in consts.GAMMAS:
    #     print(f"Gamma: {value}")
    #     call_comparison(original_image, poisson_result_holder, noise_type = "poisson", gamma = value)

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
    