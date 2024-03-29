import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras import models, layers
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

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

def compile_model(images: np.ndarray, values: np.ndarray):
    # scale values
    scaler = MinMaxScaler()
    values_scaled = scaler.fit_transform(values.reshape(-1, 1))

    # divide data to test and train sets
    train_images, test_images, train_labels, test_labels = train_test_split(images, values_scaled, test_size=0.2, random_state=42)
    input_shape = train_images[0].shape
    
    # create the model
    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
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

    print('train...\n')
    history = model.fit(train_images, train_labels, epochs=3, batch_size=32, verbose=1)

    # TODO: plotting accuracy, loss, or other information about the history

    score = model.evaluate(test_images, test_labels)
    print('Test Loss:', score[0])
    print('Test accuracy:', score[1])

    predictions = model.predict(test_images, batch_size=32, verbose=1)
    # show 200 predictions for reference...
    for i in range (200):
        print(f"prediction: {predictions[i]} real value: {test_labels[i]}")

    # plot predicions and correct values
    plt.figure(figsize=(10,10),)
    plt.scatter(test_labels, predictions, c='crimson', alpha=0.5)

    p1 = max(max(predictions), max(test_labels))
    p2 = min(min(predictions), min(test_labels))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('True Values', fontsize=10)
    plt.ylabel('Predictions', fontsize=10)
    plt.axis('equal')
    plt.show()