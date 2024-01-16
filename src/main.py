import numpy as np
import utils.image_tools as image_tools
import utils.noise_tools as noise_tools
import cv2
import metrics.metrics_caller as mc
import metrics.metrics_comparisons as mcc

def main():
    original_image = image_tools.load_image('../assets/cat.jpg')
    noisy_image = noise_tools.salt_and_pepper(original_image, 12000)

    # MSE with original image and its rotated ones
    mcc.mse_comparison(original_image)

    # ERGAS with original image and its rotated ones
    mcc.ergas_comparison(original_image)

    # PSNR with original image and its rotated ones
    mcc.psnr_comparison(original_image)

    # PSNR with original image and its rotated ones
    mcc.ssim_comparison(original_image)

    # MSE with original image and its rotated ones with (default - 12k) noise
    mcc.mse_comparison(original_image, True)

    # ERGAS with original image and its rotated ones with (default - 12k) noise
    mcc.ergas_comparison(original_image, True)

    # PSNR with original image and its rotated ones with (default - 12k) noise
    mcc.psnr_comparison(original_image, True)

    # SSIM with original image and its rotated ones with (default - 12k) noise
    mcc.ssim_comparison(original_image, True)

    # salt-pepper noise
    # image_tools.show_image("noised", noise_tools.salt_and_pepper(original_image, 16000))




if __name__ == "__main__":
    main()
    