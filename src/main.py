import numpy as np
import utils.image_tools as image_tools
import utils.noise_tools as noise_tools
import cv2
import metrics.metrics_caller as mc
import metrics.metrics_comparisons as mcc

def main():
    original_image = image_tools.load_image('../assets/cat.jpg')
    # noisy_image = noise_tools.salt_and_pepper(original_image, 20000)

    # original
    call_comparison(original_image)

    print("----------------------------")

    # salt&pepper
    call_comparison(original_image, True, "salt&pepper")

    print("----------------------------")

    # gaussian
    call_comparison(original_image, True, "gaussian")

    print("----------------------------")

    # poisson
    call_comparison(original_image, True, "poisson")

    # ----------------------------
    print("----------------------------")

    # image_tools.show_image("resized", cv2.resize(original_image, (400, 230)))
    # image_tools.show_image("resized", cv2.resize(original_image, (1890, 1700)))


def call_comparison(original_image: np.ndarray, useNoise: bool = False, noiseType: str = "") -> None:
    if noiseType == "":
        print("rotated comparison")
        image_tools.show_image("rotated", image_tools.generate_180_rotated(original_image))
    elif noiseType == "salt&pepper":
        print("salt&pepper noise comparison (default - 20k)")
        image_tools.show_image("salt&pepper", noise_tools.salt_and_pepper(original_image, 20000))
    elif noiseType == "gaussian":
        print("gaussian noise comparison (default - mean 0.5, sigma 200")
        image_tools.show_image("gaussian", noise_tools.gaussian(original_image, 0.5, 200))
    elif noiseType == "poisson":
        print("poisson noise comparison (default - 0.2 gamma)")
        image_tools.show_image("poisson", noise_tools.poisson(original_image, 0.2))

    # MSE
    mcc.mse_comparison(original_image, useNoise, noiseType)

    # ERGAS
    mcc.ergas_comparison(original_image, useNoise, noiseType)

    # PSNR
    mcc.psnr_comparison(original_image, useNoise, noiseType)

    # SSIM
    mcc.ssim_comparison(original_image, useNoise, noiseType)

    # MS-SSIM
    mcc.msssim_comparison(original_image, useNoise, noiseType)

    # VIF
    mcc.vif_comparison(original_image, useNoise, noiseType)

    # SCC
    mcc.scc_comparison(original_image, useNoise, noiseType)

    # SAM
    mcc.sam_comparison(original_image, useNoise, noiseType)


if __name__ == "__main__":
    main()
    