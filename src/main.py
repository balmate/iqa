import utils.data_tools as dt
import utils.image_tools as it

def main():

    # metric testing
    it.image_processing()

    # model trainings
    # metrics one by one
    dt.model_processing_one_by_one()
    # using all metrics together
    dt.compile_model_with_all_metrics()
    # using groups of metrics
    dt.model_processing_by_groups()


if __name__ == "__main__":
    main()
    