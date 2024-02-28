class ResultHolder:

    def __init__(self, name: str) -> None:
        self.name = name
        self.mse_results = list()
        self.ergas_results = list()
        self.psnr_results = list()
        self.ssim_results = list()
        self.ms_ssim_results = list()
        self.vif_results = list()
        self.scc_results = list()
        self.sam_results = list()