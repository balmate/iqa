import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras import models, layers
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
import classes.MetricHolder as MetricHolder

def create_data():
    # read both csvs data
    all_values = concat_data(read_data('./csvs/dmos.csv', 'dmos', True), read_data('./csvs/csiq_dmos.csv', 'dmos'))
    return all_values

def read_data(file_path: str, col: str, normalize: bool = False) -> pd.Series:
    # read concrete csv's dmos col, skip bad lines
    values = pd.read_csv(file_path, on_bad_lines='skip')[col]
    return np.array(values)

def kadid_data():
    # return kadid csvs dmos data
    return read_data('./csvs/dmos.csv', 'dmos')

def concat_data(kadid_values: pd.Series, csiq_values: pd.Series) -> pd.Series:
    return pd.concat([kadid_values, csiq_values])

def convert_to_dataframe_and_save_to_csv(data: MetricHolder.MetricHolder):
    df = pd.DataFrame({'': data.metric_values[0]})

def compile_model(images: np.ndarray, values: np.ndarray, metric: str):
    # scale values
    print("Scaleing label values...")
    scaler = MinMaxScaler()
    values_scaled = scaler.fit_transform(values.reshape(-1, 1))

    for i in range(len(values_scaled)):
        if math.isnan(values_scaled[i]):
            values_scaled[i] = 0.0
    
    # for i in range(10):
    #     print (values_scaled[i])

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
    
    print('compile...\n')
    model.compile(optimizer='adam',
                loss='mse',
                metrics=['mae'])

    print(f'train by {metric} metric...\n')
    train_images = train_images.astype(np.float16)
    history = model.fit(train_images, train_labels, epochs=3, batch_size=64, verbose=1)

    # TODO: plotting accuracy, loss, or other information about the history

    print("Evaluating model...")
    score = model.evaluate(test_images, test_labels)
    print('Test Loss:', score[0])
    print('Test accuracy:', score[1])

    print("Predicting...")
    predictions = model.predict(test_images, verbose=1)
    # show 200 predictions for reference...
    # for i in range (50):
    #     print(f"prediction: {predictions[i]} real value: {test_labels[i]}")

    # plot predicions and correct values
    plt.figure(figsize=(10,10),)
    plt.scatter(test_labels, predictions, c='crimson', alpha=0.5)

    # p1 = max(max(predictions), max(test_labels))
    # p2 = min(min(predictions), min(test_labels))
    # if hasattr(p1, "__len__"): p1 = p1[0]
    # if hasattr(p2, "__len__"): p2 = p2[0]
    # print(f"p1: {p1}, p2: {p2}")
    plt.plot([0, 1], [0, 1], 'b-')
    plt.xlabel('True Values', fontsize=10)
    plt.ylabel('Predictions', fontsize=10)
    plt.axis('equal')
    plt.title(metric)
    # plt.show()
    plt.savefig(f"ai_results/{metric}_result.jpg")


def compile_model_with_values(values: np.array, labels: np.array):

    train_values, test_values, train_labels, test_labels = train_test_split(values, labels, test_size=0.15, random_state=42)

    # create the model
    model = models.Sequential()
    model.add(layers.Dense(256, input_dim=3, activation='linear'))
    model.add(layers.Dense(256, activation='linear'))
    model.add(layers.Dense(1, activation='linear'))

    model.summary()

    # compile model
    print('compile...\n')
    model.compile(optimizer='adam',
                loss='mse',
                metrics=['mae'])
    
    history = model.fit(train_values, train_labels, epochs=100, batch_size=64, verbose=1)

    # evaluate model
    score = model.evaluate(test_values, test_labels, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    print("Predicting...")
    predictions = model.predict(test_values, verbose=1)
    # show 200 predictions for reference...
    # for i in range (50):
    #     print(f"prediction: {predictions[i]} real value: {test_labels[i]}")

    # plot predicions and correct values
    plt.figure(figsize=(10,10),)
    plt.scatter(test_labels, predictions, c='crimson', alpha=0.5)

    # p1 = max(max(predictions), max(test_labels))
    # p2 = min(min(predictions), min(test_labels))
    # if hasattr(p1, "__len__"): p1 = p1[0]
    # if hasattr(p2, "__len__"): p2 = p2[0]
    # print(f"p1: {p1}, p2: {p2}")
    plt.plot([0, 5], [0, 5], 'b-')
    plt.xlabel('True Values', fontsize=10)
    plt.ylabel('Predictions', fontsize=10)
    plt.axis('equal')
    plt.title("mse, ergas, psnr")
    plt.savefig("ai_results/mse_ergas_psnr_result.jpg")
    plt.show()