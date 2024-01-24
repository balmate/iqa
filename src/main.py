import numpy as np
import utils.image_tools as image_tools
import utils.noise_tools as noise_tools
import metrics.metrics_comparisons as mcc

def main():
    original_image = image_tools.load_image('../assets/nature.jpg')
    image_tools.show_image("original", original_image)
    
    # scaled_down = image_tools.create_scaled_image(original_image, 30)
    # scaled_up = image_tools.create_scaled_image(original_image, 170)

    # rotated
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

    print("----------------------------")


def call_comparison(image: np.ndarray, use_noise: bool = False, noise_type: str = "") -> None:
    if noise_type == "":
        print("rotated comparison")
        image_tools.show_image("rotated", image_tools.generate_180_rotated(image))
    elif noise_type == "salt&pepper":
        print("salt&pepper noise comparison (default - 15k)")
        image_tools.show_image("salt&pepper", noise_tools.salt_and_pepper(image, 15000))
    elif noise_type == "gaussian":
        print("gaussian noise comparison (default - mean 0.5, sigma 100)")
        image_tools.show_image("gaussian", noise_tools.gaussian(image, 0.5, 100))
    elif noise_type == "poisson":
        print("poisson noise comparison (default - 0.5 gamma)")
        image_tools.show_image("poisson", noise_tools.poisson(image, 0.5))

    # MSE
    mcc.mse_comparison(image, use_noise, noise_type)

    # ERGAS
    mcc.ergas_comparison(image, use_noise, noise_type)

    # PSNR
    mcc.psnr_comparison(image, use_noise, noise_type)

    # SSIM
    mcc.ssim_comparison(image, use_noise, noise_type)

    # MS-SSIM
    mcc.msssim_comparison(image, use_noise, noise_type)

    # VIF
    mcc.vif_comparison(image, use_noise, noise_type)

    # SCC
    mcc.scc_comparison(image, use_noise, noise_type)

    # SAM
    mcc.sam_comparison(image, use_noise, noise_type)


if __name__ == "__main__":
    main()
    