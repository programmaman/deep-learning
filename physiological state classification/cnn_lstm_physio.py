import sys
import time
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import create_dataset

# Some code from Dr. Canavan's sample code: saveLoadModel.py, evaluationMetrics.py

# Parse Arguments
if len(sys.argv) < 5:
    print("Please enter arguments: 'train' or 'test, "
          "path to the Physiological data folder, model name, "
          "data modal type.")

cmd_path = sys.argv[2]
model_name = sys.argv[3]
data_modal = sys.argv[4].upper()


def test_train(cmd):
    if cmd == 'train':
        path = cmd_path + "/" + "Training"
        val_path = cmd_path + "/" + "Validation"
        return train(model_name, path, val_path)
    if cmd == 'test':
        path = cmd_path + "/" + "Testing"
        return test(model_name, path)
    raise Exception('Enter train OR test as your second argument')


# Configuration Parameters
act = 'relu'


def train(name, path, val_path):
    print("Loading Training Data: ")
    time.sleep(1)
    trainingData, trainingLabels = create_dataset.create_dataset(path, data_modal)
    trainingData = tf.keras.preprocessing.sequence.pad_sequences(trainingData, 1000, dtype=np.float32)
    trainingLabels = tf.convert_to_tensor(trainingLabels, dtype=tf.int64)

    print("Loading Validation Data: ")
    time.sleep(1)
    validationData, validationLabels = create_dataset.create_dataset(val_path, data_modal)
    validationData = tf.keras.preprocessing.sequence.pad_sequences(validationData, 1000, dtype=np.float32)
    validationLabels = tf.convert_to_tensor(validationLabels, dtype=tf.int64)

    print("Initializing Neural Network.")
    time.sleep(2)

    model = create_model()
    checkpoint = [tf.keras.callbacks.ModelCheckpoint(filepath="./models/" + name + ".h5",
                                                     monitor='val_sparse_categorical_accuracy',
                                                     verbose=1,
                                                     save_best_only=True,
                                                     mode='max'),
                  tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy',
                                                   min_delta=.01,
                                                   patience=5)]

    model.fit(trainingData, trainingLabels,
              validation_data=(validationData, validationLabels), epochs=25, callbacks=checkpoint)
    savedModel = tf.keras.models.load_model("./models/" + name + ".h5")
    output_metrics(savedModel, validationData, validationLabels)


def test(name, path):
    testingData, testingLabels = create_dataset.create_dataset(path, data_modal)
    testingData = tf.keras.preprocessing.sequence.pad_sequences(testingData, 1000, dtype=np.float32)
    testingLabels = tf.convert_to_tensor(testingLabels, dtype=tf.int64)
    model = tf.keras.models.load_model("./models/" + name + ".h5")
    model.evaluate(testingData, testingLabels)
    output_metrics(model, testingData, testingLabels)


# Neural Network
def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Masking(mask_value=0., input_shape=(1000, 1)))
    model.add(layers.Conv1D(4, kernel_size=7, activation='tanh'))  # Slide window of the temporal data
    model.add(layers.Conv1D(8, kernel_size=5, activation='tanh'))  # Slide window of the temporal data
    model.add(layers.Conv1D(16, kernel_size=3, activation='relu'))  # Slide window of the temporal data
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.LSTM(10, return_sequences=True))
    model.add(layers.Dense(128))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


# Output Recall (Macro/Micro), Precision (Macro/Micro), F1 Score (Macro/Micro) and Confusion Matrix
def output_metrics(model, data, labels):
    # predict and format output to use with sklearn
    predict = model.predict(data)
    print(np.shape(predict))
    predict = np.argmax(predict, axis=1)
    print(np.shape(predict))
    # macro precision and recall
    precisionMacro = precision_score(labels, predict, average='macro')
    recallMacro = recall_score(labels, predict, average='macro')
    # micro precision and recall
    precisionMicro = precision_score(labels, predict, average='micro')
    recallMicro = recall_score(labels, predict, average='micro')
    confMat = confusion_matrix(labels, predict)
    print("Macro precision: ", precisionMacro)
    print("Micro precision: ", precisionMicro)
    print("Macro recall: ", recallMacro)
    print("Micro recall: ", recallMicro)
    print(confMat)


test_train(sys.argv[1])
