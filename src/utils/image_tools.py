import os
import cv2
import numpy as np
import pandas as pd
from classes.ResultHolder import ResultHolder
import utils.noise_tools as noise_tools
import imutils
from keras.preprocessing.image import img_to_array
import metrics.metrics_comparisons as mc
import classes.MetricHolder as MetricHolder
import pandas as pd
from utils import consts
import os
import utils.plotting_tools as pt

def load_image(image_to_load: str, image_name: str = "test") -> np.ndarray:
    '''
    Load an image from the project folder.
    Optionally set a name to the image for printing purposes.
    Use the image and the name as global variables to be able to use them in other functions.
    '''
    original = cv2.imread(image_to_load)
    global original_image
    global current_image_name
    current_image_name = image_name
    original_image = original
    return original

def generate_180_rotated_with_noise(image_to_modify: np.ndarray, noise_type: str, number_of_pixels_to_transform: int = 15000, mean: float = 0.5, sigma: float = 100,
                                     gamma: float = 0.5, blur: list = (5,5), fade_percent: float = 0.2, saturation: float = 0.2, alpha: float = 0.5, zoom: float = 1.5):
    '''
    Generate a specific noisy image and rotate it by 180 degs.
    '''
    noisy_image = create_concrete_noisy_image(image_to_modify, noise_type, number_of_pixels_to_transform, mean, sigma, gamma, blur, fade_percent, saturation, alpha, zoom)
    return cv2.rotate(noisy_image, cv2.ROTATE_180)

def show_image(window_title: str, image: np.ndarray) -> None:
    '''
    Display the iamge being processed.
    Currently no need for display, a return statement inserted to the begining of the function.
    '''
    return
    cv2.imshow(window_title, image)
    cv2.waitKey(0)

def create_scaled_image(original_image: np.ndarray, scale_value: int):
    '''
    Scale the passed image by the passed value.
    '''
    new_size = (int(original_image.shape[1] * scale_value / 100), int(original_image.shape[0] * scale_value / 100))
    return cv2.resize(original_image, new_size, interpolation=cv2.INTER_AREA)

def create_concrete_noisy_image(image_to_return: np.ndarray, noise_type: str, number_of_pixels_to_transform: int, mean: float, sigma: float, gamma: float, blur: list,
                                 fade_percent: float, saturation: float, alpha: float, zoom: float):
    '''
    Create a concrete noised image by the parameters.
    The noise_type param tells the function what noise should it apply on the image.
    Then use the proper params to create the noisy image.
    '''
    if noise_type == "salt&pepper":
        return noise_tools.salt_and_pepper(image_to_return, number_of_pixels_to_transform)
    elif noise_type == "gaussian":
        return noise_tools.gaussian(image_to_return, mean, sigma)
    elif noise_type == "poisson":
        return noise_tools.poisson(image_to_return, gamma)
    elif noise_type == "blur":
        return noise_tools.blur(image_to_return, blur)
    elif noise_type == "fade":
        return noise_tools.fade(image_to_return, fade_percent)
    elif noise_type == "saturation":
        return noise_tools.saturation(image_to_return, saturation)
    elif noise_type == "contrast":
        return noise_tools.contrast(image_to_return, alpha)
    elif noise_type == "zoom":
        return noise_tools.zoom(image_to_return, zoom)
    
def create_rotated_image(image: np.ndarray, angle: int) -> np.ndarray:
  '''
  Create rotated image by the passed rotation angle.
  '''
  return imutils.rotate(image, angle)

def save_image(image: np.ndarray, path: str) -> None:
    '''
    Save the image to the specified path.
    '''
    # print("path: " + path)
    cv2.imwrite(path, image)


def image_processing():
    '''
    The main function that creates noisy images, calculates metric values and plots the results.
    Uses the first 5 kadid reference images.
    '''
    path_to_images = "../assets/kadid_ref_images"

    for image_file in os.listdir(path_to_images):
        image_path = os.path.join(path_to_images, image_file)

        image_name = image_file.split('.')[0]
        image = load_image(image_path, image_name)
        show_image(image_name, image)

        # create result holders for plotting
        rotation_result_holder = ResultHolder("rotations", image_name)
        salt_pepper_result_holder = ResultHolder("salt&pepper", image_name)
        gaussian_result_holder = ResultHolder("gaussian", image_name)
        blur_result_holder = ResultHolder("blur", image_name)
        fade_result_holder = ResultHolder("fade", image_name)
        saturation_low_result_holder = ResultHolder("saturation_low", image_name)
        saturation_high_result_holder = ResultHolder("saturation_high", image_name)
        contrast_dark_result_holder = ResultHolder("contrast_dark", image_name)
        contrast_light_result_holder = ResultHolder("contrast_light", image_name)
        zoom_result_holder = ResultHolder("zoom", image_name)

        # original vs zoomed
        print("Comparison: zoom with different param values")
        for zoom in consts.ZOOM_VALUES:
            print(f"Zoom value: {zoom}")
            mc.call_comparison(image, zoom_result_holder, noise_type = "zoom", zoom = zoom)

        # plotting the results
        pt.create_plots_from_object(zoom_result_holder, consts.ZOOM_VALUES, "zoom values", "zoom")

        # original vs saturated
        print("Comparison: contrast (dark) with different param values")
        for alpha in consts.CONTRAST_DARK:
            print(f"Contrast value: {alpha}")
            mc.call_comparison(image, contrast_dark_result_holder, noise_type = "contrast", alpha = alpha)

        # plotting the results
        pt.create_plots_from_object(contrast_dark_result_holder, consts.CONTRAST_DARK, "contrast (dark) values", "contrast_dark")

        # original vs saturated
        print("Comparison: contrast (light) with different param values")
        for alpha in consts.CONTRAST_LIGHT:
            print(f"Contrast value: {alpha}")
            mc.call_comparison(image, contrast_light_result_holder, noise_type = "contrast", alpha = alpha)

        # plotting the results
        pt.create_plots_from_object(contrast_light_result_holder, consts.CONTRAST_LIGHT, "contrast (light) values", "contrast_light")

        # original vs saturated
        print("Comparison: saturation (low) with different param values")
        for percent in consts.SATURATION_LOW:
            print(f"Saturation (low) value: {percent}")
            mc.call_comparison(image, saturation_low_result_holder, noise_type = "saturation", saturation = percent)

        # plotting the results
        pt.create_plots_from_object(saturation_low_result_holder, consts.SATURATION_LOW, "saturation (low) values", "saturation_low")

        # original vs saturated
        print("Comparison: saturation (high) with different param values")
        for percent in consts.SATURATION_HIGH:
            print(f"Saturation (high) value: {percent}")
            mc.call_comparison(image, saturation_high_result_holder, noise_type = "saturation", saturation = percent)

        # plotting the results
        pt.create_plots_from_object(saturation_high_result_holder, consts.SATURATION_HIGH, "saturation (high) values", "saturation_high")

        # original vs faded
        print("Comparison: fade with different param values")
        for percent in consts.FADE_VALUES:
            print(f"Fade value: {percent}")
            mc.call_comparison(image, fade_result_holder, noise_type = "fade", fade_percent = percent)

        # plotting the results
        pt.create_plots_from_object(fade_result_holder, consts.FADE_VALUES, "fade values", "faded")
            
        # original vs blurred
        print("Comparison: blur with different param values")
        for ksize in consts.KERNEL_SIZES:
            print(f"Blur value: {ksize}")
            mc.call_comparison(image, blur_result_holder, noise_type = "blur", blur = ksize)
        
        # plotting the results
        pt.create_plots_from_object(blur_result_holder, consts.KERNEL_SIZES, "blur values", "blurred")

        # original vs rotated with different angles
        print("Comparison: rotated with different angles")
        for angle in consts.ANGLES:
            print(f"Angle: {angle} degrees")
            mc.call_comparison(image, rotation_result_holder, True, angle)
        
        # plotting the results
        pt.create_plots_from_object(rotation_result_holder, consts.ANGLES, "angles (in degree)", "rotation", None, None)

        # original vs salt&pepper with different pixel values to transform
        print("Comparison: salt&pepper with different param values")
        for value in consts.PIXELS_TO_TRANSFORM:
            print(f"Number of pixels to transform: {value}")
            mc.call_comparison(image, salt_pepper_result_holder, noise_type = "salt&pepper", number_of_pixels_to_transform = value)
        
        # plotting the results
        pt.create_plots_from_object(salt_pepper_result_holder, consts.PIXELS_TO_TRANSFORM, "pixels transformed", "salt&pepper")

        # original vs gaussian with different param values
        print("Comparison: gaussian with different param values")
        for i in range(5):
            print(f"Mean: {consts.MEANS[i]}, sigma: {consts.SIGMAS[i]}")
            mc.call_comparison(image, gaussian_result_holder, noise_type = "gaussian", mean = consts.MEANS[i], sigma = consts.SIGMAS[i])

        # plotting the results
        pt.create_plots_from_object(gaussian_result_holder, consts.MEANS, "mean", "gaussian", consts.SIGMAS, "sigma")

def get_kadid_images():
    # ONLY FOR LOCALE TESTING (with all of the images the source would be too big)
    path_to_images = 'C:\images'
    images = []
    for image_file in os.listdir(path_to_images):
        image_path = os.path.join(path_to_images, image_file)
        # image = img_to_array(load_img(image_path, color_mode='rgb', target_size=(192, 256)), dtype=np.uint8) / 255.0 # it worked with this
        # load as ndarray
        image = img_to_array(load_image(image_path), dtype=np.uint8)
        # rescale image, and normalize it
        image = img_to_array(cv2.resize(image, (256, 192), interpolation=cv2.INTER_AREA), dtype=np.uint8) / 255.0
        images.append(image)

    return np.array(images)

def get_kadid_images_with_metric_values(metric: str) -> MetricHolder.MetricHolder:
    # ONLY FOR LOCALE TESTING (with all of the images the source would be too big)
    # metrics = ["mse", "ergas", "psnr", "ssim", "ms-ssim", "vif", "scc", "sam"]
    path_to_images = 'C:\images'
    dmos_values = pd.read_csv("./csvs/dmos.csv", on_bad_lines='skip')["dmos"]
    data = MetricHolder.MetricHolder()
    i = 0
    df = pd.DataFrame(columns=[metric])
    print(f"Reading images and values for metric: {metric}...")
    for image_file in os.listdir(path_to_images):
        image_path = os.path.join(path_to_images, image_file)
        # image = img_to_array(load_img(image_path, color_mode='rgb', target_size=(192, 256)), dtype=np.uint8) / 255.0 # it worked with this
        # load as ndarray
        image = img_to_array(load_image(image_path), dtype=np.uint8)
        # rescale image, and normalize it
        # get the original image
        original_image_path = "../assets/kadid_ref_images/" + image_file.split('_')[0] + ".png"
        # data.metric_values.append([mc.call_prints(load_image(original_image_path), image, "mse", "original"), 
        #                            mc.call_prints(load_image(original_image_path), image, "ergas", "original"),
        #                            mc.call_prints(load_image(original_image_path), image, "psnr", "original")])
        print("metric val added")
        metric_val = mc.call_prints(load_image(original_image_path), image, metric, "original")
        data.dmos.append(dmos_values[i])
        # print(f"metric values: {data.metric_values[i]}")
        # print(f"dmos value: {data.dmos[i]}")
        df = df._append({metric: metric_val}, ignore_index=True)
        i += 1
    print(f"{metric} values:")
    # print(df)
    # print(f"saving data to {metric} file")
    # df.to_csv(f"csvs/{metric}_values.csv", mode='a', header=False)
    return data
