import matplotlib.pyplot as plt
from classes.ResultHolder import ResultHolder
import inspect

def create_plot_for_metric_simple_param(x: list, labelx: str, y: list, labely: str, title: str, file_name: str) -> None:
    # data
    print(f"plotting method: {title}\n{x}\n{y}\n")
    plt.figure()
    plt.plot(x, y, "o-g")
    
    # axis names and title
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.title(title)

    # show
    plt.legend()
    # plt.show()

    # save
    save_plot(file_name)

def create_plot_for_metric_double_param(x: list, labelx: str, y: list, labely: str, z: list, labelz: str,  title: str, file_name: str):
    print(f"plotting 3d method: {title}\n{x}\n{y}\n{z}\n")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # data
    ax.plot(x, y, z, "o-g")

    # axis names and title
    ax.set_title(title)
    ax.set_xlabel(labelx)
    ax.set_ylabel(labely)
    ax.set_zlabel(labelz)
    
    # show
    ax.legend()
    # plt.show()

    # save
    save_plot(file_name)

def create_plots_from_object(result_holder: ResultHolder, param_consts: list, param_consts_name: str, transform: str, second_param_consts: list = None, second_param_consts_name: str = None):
    for i in inspect.getmembers(result_holder):
        if second_param_consts is None:
            if i[0] == "mse_results":
                create_plot_for_metric_simple_param(param_consts, param_consts_name, result_holder.mse_results, "mse values", "mse with " + transform, f"{result_holder.image_name}_mse_{transform}")
            elif i[0] == "ergas_results":
                create_plot_for_metric_simple_param(param_consts, param_consts_name, result_holder.ergas_results, "ergas values", "ergas with " + transform, f"{result_holder.image_name}_ergas_{transform}")
            elif i[0] == "psnr_results":
                create_plot_for_metric_simple_param(param_consts, param_consts_name, result_holder.psnr_results, "psnr values", "psnr with " + transform, f"{result_holder.image_name}_psnr_{transform}")
            elif i[0] == "ssim_results":
                create_plot_for_metric_simple_param(param_consts, param_consts_name, result_holder.ssim_results, "ssim values", "ssim with " + transform, f"{result_holder.image_name}_ssim_{transform}")
            elif i[0] == "ms_ssim_results":
                create_plot_for_metric_simple_param(param_consts, param_consts_name, result_holder.ms_ssim_results, "ms-ssim values", "ms-ssim with " + transform, f"{result_holder.image_name}_ms_ssim_{transform}")
            elif i[0] == "vif_results":
                create_plot_for_metric_simple_param(param_consts, param_consts_name, result_holder.vif_results, "vif values", "vif with " + transform, f"{result_holder.image_name}_vif_{transform}")
            elif i[0] == "scc_results":
                create_plot_for_metric_simple_param(param_consts, param_consts_name, result_holder.scc_results, "scc values", "scc with " + transform, f"{result_holder.image_name}_scc_{transform}")
            elif i[0] == "sam_results":
                create_plot_for_metric_simple_param(param_consts, param_consts_name, result_holder.sam_results, "sam values", "sam with " + transform, f"{result_holder.image_name}_sam_{transform}")
        else:
            if i[0] == "mse_results":
                create_plot_for_metric_double_param(param_consts, param_consts_name, second_param_consts, second_param_consts_name, 
                                                    result_holder.mse_results, "mse values", "mse with " + transform, f"{result_holder.image_name}_mse_{transform}")
            elif i[0] == "ergas_results":
                create_plot_for_metric_double_param(param_consts, param_consts_name, second_param_consts, second_param_consts_name, 
                                                    result_holder.ergas_results, "ergas values", "ergas with " + transform, f"{result_holder.image_name}_ergas_{transform}")
            elif i[0] == "psnr_results":
                create_plot_for_metric_double_param(param_consts, param_consts_name, second_param_consts, second_param_consts_name, 
                                                    result_holder.psnr_results, "psnr values", "psnr with " + transform, f"{result_holder.image_name}_psnr_{transform}")
            elif i[0] == "ssim_results":
                create_plot_for_metric_double_param(param_consts, param_consts_name, second_param_consts, second_param_consts_name, 
                                                    result_holder.ssim_results, "ssim values", "ssim with " + transform, f"{result_holder.image_name}_ssim_{transform}")
            elif i[0] == "ms_ssim_results":
                create_plot_for_metric_double_param(param_consts, param_consts_name, second_param_consts, second_param_consts_name, 
                                                    result_holder.ms_ssim_results, "ms-ssim values", "ms-ssim with " + transform, f"{result_holder.image_name}_ms_ssim_{transform}")
            elif i[0] == "vif_results":
                create_plot_for_metric_double_param(param_consts, param_consts_name, second_param_consts, second_param_consts_name, 
                                                    result_holder.vif_results, "vif values", "vif with " + transform, f"{result_holder.image_name}_vif_{transform}")
            elif i[0] == "scc_results":
                create_plot_for_metric_double_param(param_consts, param_consts_name, second_param_consts, second_param_consts_name, 
                                                    result_holder.scc_results, "scc values", "scc with " + transform, f"{result_holder.image_name}_scc_{transform}")
            elif i[0] == "sam_results":
                create_plot_for_metric_double_param(param_consts, param_consts_name, second_param_consts, second_param_consts_name, 
                                                    result_holder.sam_results, "sam values", "sam with " + transform, f"{result_holder.image_name}_sam_{transform}")

def save_plot(file_name: str) -> None:
    metric = file_name.split('_')[1]
    if metric == "ms":
        plt.savefig("results/ms_ssim/" + file_name)
    else:
        plt.savefig("results/" + file_name.split('_')[1] + "/" + file_name)