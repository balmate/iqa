import numpy as np
import utils.image_tools as image_tools
import utils.noise_tools as noise_tools
import cv2
import metrics.metrics_caller as mc
import metrics.metrics_comparisons as mcc

def main():
    original_image = image_tools.load_image('../assets/cat.jpg')
    # noisy_image = noise_tools.salt_and_pepper(original_image, 20000)

    # MSE with original image and its rotated ones
    mcc.mse_comparison(original_image)

    # ERGAS with original image and its rotated ones
    mcc.ergas_comparison(original_image)

    # PSNR with original image and its rotated ones
    mcc.psnr_comparison(original_image)

    # SSIM with original image and its rotated ones
    mcc.ssim_comparison(original_image)

    # MS-SSIM with original image and its rotated ones
    mcc.msssim_comparison(original_image)

    # VIF with original image and its rotated ones
    mcc.vif_comparison(original_image)

    # SCC with original image and its rotated ones
    mcc.scc_comparison(original_image)

    # SAM with original image and its rotated ones
    mcc.sam_comparison(original_image)

    # ----------------------------
    print("----------------------------")

    # MSE with original image and its rotated ones with (default - 20k) noise
    mcc.mse_comparison(original_image, True)

    # ERGAS with original image and its rotated ones with (default - 20k) noise
    mcc.ergas_comparison(original_image, True)

    # PSNR with original image and its rotated ones with (default - 20k) noise
    mcc.psnr_comparison(original_image, True)

    # SSIM with original image and its rotated ones with (default - 20k) noise
    mcc.ssim_comparison(original_image, True)
    
    # MS-SSIM with original image and its rotated ones with (default - 20k) noise
    mcc.msssim_comparison(original_image, True)

    # VIF with original image and its rotated ones with (default - 20k) noise
    mcc.vif_comparison(original_image, True)

    # SCC with original image and its rotated ones with (default - 20k) noise
    mcc.scc_comparison(original_image, True)

    # SAM with original image and its rotated ones with (default - 20k) noise
    mcc.sam_comparison(original_image, True)

    # salt-pepper noise
    image_tools.show_image("salt&pepper noised", noise_tools.salt_and_pepper(original_image, 16000))

    # gauss noise
    image_tools.show_image("gaussian noised", noise_tools.gaussian(original_image))

    # poisson noise
    image_tools.show_image("poisson noised", noise_tools.poisson(original_image, 0.2))


if __name__ == "__main__":
    main()
    