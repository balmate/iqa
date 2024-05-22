from decimal import Decimal
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras import models, layers, optimizers
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import classes.MetricHolder as MetricHolder
from matplotlib.patches import Rectangle

def read_data(file_path: str, col: str, normalize: bool = False) -> pd.Series:
    '''
    Reads the specified data column from the specified csv file, and return it as a numpy array.
    '''
    # read concrete csv's dmos col, skip bad lines
    values = pd.read_csv(file_path, on_bad_lines='skip')[col]
    return np.array(values)

def kadid_data():
    '''
    Reads all the DMOS scores from the kadid csv file.
    '''
    # return kadid csvs dmos data
    return read_data('./csvs/dmos.csv', 'dmos')

def get_metric_values_from_csv(metric: str) -> pd.Series:
    '''
    Reads all the calculated metric values from the specific csv file.
    '''
    return read_data(f"./csvs/{metric}_values.csv", metric)

def compile_model(images: np.ndarray, values: np.ndarray, metric: str):
    '''
    Creates, compiles and evaluates a convolution neural network by the transformed images of the kadid dataset.
    '''
    # scale values
    print("Scaleing label values...")
    scaler = MinMaxScaler()
    values_scaled = scaler.fit_transform(values.reshape(-1, 1))

    # divide data to test and train sets
    print("Splitting data to train and test sets...")
    train_images, test_images, train_labels, test_labels = train_test_split(images, values_scaled, test_size=0.2, random_state=42)
    input_shape = train_images[0].shape
    
    # create the model
    model = models.Sequential()
    model.add(layers.Input(input_shape))
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))
    model.summary()
    
    # compile the model
    print('compile...\n')
    model.compile(optimizer='adam',
                loss='mse',
                metrics=['mae'])

    # train the model
    print(f'train by {metric} metric...\n')
    train_images = train_images.astype(np.float16)
    history = model.fit(train_images, train_labels, epochs=3, batch_size=64, verbose=1)

    # evaluating model
    print("Evaluating model...")
    score = model.evaluate(test_images, test_labels)
    print('Test Loss:', score[0])
    print('Test accuracy:', score[1])

    # predictions
    print("Predicting...")
    predictions = model.predict(test_images, verbose=1)

    # plot predicions and correct values
    plt.figure(figsize=(10,10),)
    plt.scatter(test_labels, predictions, c='crimson', alpha=0.5)

    plt.plot([0, 1], [0, 1], 'b-')
    plt.xlabel('True Values', fontsize=10)
    plt.ylabel('Predictions', fontsize=10)
    plt.axis('equal')
    plt.title(metric)
    # plt.show()
    plt.savefig(f"ai_results/{metric}_result.jpg")


def compile_concrete_model_with_values(values: np.array, labels: np.array, metric: str, input_dim: int = 1):
    '''
    Creates, compiles and evaluates a dense neural network by the metric values calculated before on the kadid transformed images.
    '''
    # separate data to train and test sets
    train_values, test_values, train_labels, test_labels = train_test_split(values, labels, test_size=0.12)
    
    # normalize data
    scaler = StandardScaler()
    if input_dim > 1:
        train_values = scaler.fit_transform(train_values)
        test_values = scaler.fit_transform(test_values)
    else:
        train_values = scaler.fit_transform(train_values.reshape(-1, 1))
        test_values = scaler.fit_transform(test_values.reshape(-1, 1))

    # create the model
    model = models.Sequential()
    model.add(layers.Dense(128, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))

    model.summary()

    # compile model
    print('compile...\n')
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                loss='mae',
                metrics=['mae'])

    # train model
    history = model.fit(train_values, train_labels, epochs=70, verbose=1)

    # evaluate model
    score = model.evaluate(test_values, test_labels, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # predictions
    print(f"Predicting {metric} values...")
    predictions = model.predict(test_values, verbose=1)

    # create thresholds, then evaluate model by these (partly as a confusion matrix)
    thresholds = [0.3, 0.5, 1]
    calculate_confusion_matrix(predictions, test_labels, thresholds, metric)

    # set threshold for predictions, display this accuracy on the plot
    threshold = 0.3
    accepted_predictions = []
    for i in range(len(predictions)):
        if np.abs(predictions[i] - test_labels[i]) <= threshold:
            accepted_predictions.append(predictions[i])
    modified_accuracy = len(accepted_predictions) / len(predictions)

    # plot predicions and correct values
    plt.figure(figsize=(10,10),)
    ax = plt.gca()
    plt.scatter(test_labels, predictions, c='crimson', alpha=0.5)

    # create a rectangle to illustrate threshold area
    r = Rectangle(xy=(0.9, 1), width=0.2, height=5.7, angle=-45, rotation_point=(1,1), fill=False, alpha=1)
    ax.add_patch(r)
    plt.plot()
    plt.xlabel('True Values', fontsize=10)
    plt.ylabel('Predictions', fontsize=10)
    plt.axis('equal')
    plt.title(f"{metric}\nloss: {score[0]}\n current prediction with threshold acc: {modified_accuracy * 100}")
    plt.savefig(f"ai_results_with_values/{metric}_results.jpg")
    # plt.show()

def compile_model_with_all_metrics():
    '''
    Prepare all calculated metric values for model training.
    '''
    # normalizer
    data = MetricHolder.MetricHolder()
    # get dmos scores
    data.dmos = kadid_data()

    # get values for all metric and normalize them
    mse = np.array(get_metric_values_from_csv("mse"))
    ergas = np.array(get_metric_values_from_csv("ergas"))
    psnr = np.array(get_metric_values_from_csv("psnr"))
    ssim = np.array(get_metric_values_from_csv("ssim"))
    ms_ssim = np.array(get_metric_values_from_csv("ms-ssim"))
    vif = np.array(get_metric_values_from_csv("vif"))
    sam = np.array(get_metric_values_from_csv("sam"))
    scc = np.array(get_metric_values_from_csv("scc"))

    # check NaN values
    for i in range(10125):
        if math.isnan(mse[i]) :
            mse[i] = 0.0
        elif math.isnan(ergas[i]):
            ergas[i] = 0.0
        elif math.isnan(psnr[i]):
            psnr[i] = 0.0
        elif math.isnan(ssim[i]):
            ssim[i] = 0.0
        elif math.isnan(ms_ssim[i]):
            ms_ssim[i] = 0.0
        elif math.isnan(vif[i]):
            vif[i] = 0.0
        elif math.isnan(sam[i]):
            sam[i] = 0.0
        elif math.isnan(scc[i]):
            scc[i] = 0.0
        # append values to data holder
        data.metric_values.append([mse[i].item(), ergas[i].item(), psnr[i].item(), ssim[i].item(), ms_ssim[i].item(), vif[i].item(), sam[i].item(), scc[i].item()])

    # compile and evaluate the model
    compile_concrete_model_with_values(np.array(data.metric_values), data.dmos, "all", 8)

def compile_model_by_one_metric(metric: str):
    '''
    Prepare a specific calculated metric value for model training.
    '''
    # normalizer
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    data = MetricHolder.MetricHolder()
    # get dmos scores
    data.dmos = kadid_data()

    # get values for proper metric
    data.metric_values = scaler.fit_transform(np.array(get_metric_values_from_csv(metric)).reshape(-1, 1))

    # check for NaN values
    for i in range(len(data.metric_values)):
        if math.isnan(data.metric_values[i]):
            data.metric_values[i] = 0.0
    
    # compile and evaluate model
    compile_concrete_model_with_values(np.array(data.metric_values), data.dmos, metric)

def calculate_confusion_matrix(predictions, true_values, thresholds,metric):
    '''
    Function to calculate accepted accuracies of the model by different threshold values
    '''
    print(f"total predictions: {len(predictions)}, current metric: {metric}")
    # do fo all thresholds
    for threshold in thresholds:
        full_matches = []
        threshold_matches_below = []
        threshold_matches_all = []
        differents = []
        # check all predicted and real values
        for i in range(len(predictions)):
            rounded_pred = 0
            pred_as_double = Decimal(predictions[i].item())
            # check how many decimal places the real value have and round the prediction according to it
            if np.abs(pred_as_double.as_tuple().exponent) == 2:
                rounded_pred = round(pred_as_double, 2)
            elif np.abs(pred_as_double.as_tuple().exponent) == 1:
                rounded_pred = round(pred_as_double, 1)
            else:
                rounded_pred = round(pred_as_double)

            # check that wich array to put into the rounded value
            if rounded_pred == true_values[i]:
                full_matches.append([rounded_pred, true_values[i]])
            elif true_values[i] > rounded_pred and (true_values[i] - rounded_pred) <= threshold:
                threshold_matches_below.append([rounded_pred, true_values[i]])
            elif true_values[i] < rounded_pred and (rounded_pred - true_values[i]) <= threshold:
                threshold_matches_all.append([rounded_pred, true_values[i]])
            else:
                differents.append([rounded_pred, true_values[i]])

        accepted_accuracy = len(full_matches) + (len(threshold_matches_below) / len(predictions) * 100)
        full_accuracy_in_threshold = ((len(full_matches) + len(threshold_matches_below) + len(threshold_matches_all)) / len(predictions)) * 100
        print(f"statistics for threshold: {threshold}")
        print(f"full matches: {len(full_matches)}")
        print(f"threshold matches BELOW the true value (accepted): {len(threshold_matches_below)}")
        print(f"threshold matches ABOVE the true value: {len(threshold_matches_all)}")
        print(f"differents: {len(differents)}")
        print(f"accepted accuracy: {accepted_accuracy}%\nfull accuracy in threshold: {full_accuracy_in_threshold}%")

def model_processing_by_groups():
    '''
    Function to do the whole training process from data preparation to model evaluation by groups of metrics.
    '''
    metric_combos2 = [["mse", "ergas", "psnr"], ["ssim", "ms-ssim"], ["vif", "scc", "sam"]]
    data = MetricHolder.MetricHolder()
    data.dmos = kadid_data()
    for metrics in metric_combos2:
        if len(metrics) == 2:
            ssim_data = np.array(get_metric_values_from_csv("ssim"))
            ms_ssim_data = np.array(get_metric_values_from_csv("ms-ssim"))

            for i in range(len(ssim_data)):
                if math.isnan(ssim_data[i]): ssim_data[i] = 0.0
                elif math.isnan(ms_ssim_data[i]): ms_ssim_data[i] = 0.0
                data.metric_values.append([ssim_data[i], ms_ssim_data[i]])
        else:
            data0 = np.array(get_metric_values_from_csv(metrics[0]))
            data1 = np.array(get_metric_values_from_csv(metrics[1]))
            data2 = np.array(get_metric_values_from_csv(metrics[2]))

            for i in range(len(data0)):
                if math.isnan(data0[i]): data0[i] = 0.0
                elif math.isnan(data1[i]): data1[i] = 0.0
                elif math.isnan(data2[i]): data2[i] = 0.0
                data.metric_values.append([data0[i], data1[i], data2[i]])

        compile_concrete_model_with_values(np.array(data.metric_values), data.dmos, metrics, len(metrics))
        data.metric_values = []

def model_processing_one_by_one():
    '''
    Function to do the whole training process from data preparation to model evaluation by each metric one by one.
    '''
    metrics = ["mse", "ergas", "psnr", "ssim", "ms-ssim", "vif", "scc", "sam"]
    for metric in metrics:
        print(f"Getting data and compiling model for {metric} metric...")
        compile_model_by_one_metric(metric)