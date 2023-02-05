"""
This is code is modfied from https://keras.io/examples/vision/image_classification_with_vision_transformer/.
It will not run as is. There are variables that are not defined and imports are missing.
If you use this code, you will need to define/import.
This code is so you can have a skeleton/baseline to implement a vision transformer. It uses the functional API in Keras.
"""
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score


def test_cnn(size):
    loaded_model = tf.keras.models.load_model("./" + size + '_CNN.h5')
    _, categorical, accuracy, top_5_accuracy = loaded_model.evaluate(testingData, testingLabels)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")
    print(print(f"Categorical Accuracy: {round(categorical * 100, 2)}%"))
    output_metrics(saved_model, testingData, testingLabels)


def test_vit(size):
    loaded_model = tf.keras.models.load_model("./" + size + '_vision_transformer.h5')
    _, categorical, accuracy, top_5_accuracy = loaded_model.evaluate(testingData, testingLabels)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")
    print(print(f"Categorical Accuracy: {round(categorical * 100, 2)}%"))
    output_metrics(saved_model, testingData, testingLabels)


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
    f1score = f1_score(labels, predict, average='macro')
    print("Macro precision: ", precisionMacro)
    print("Micro precision: ", precisionMicro)
    print("Macro recall: ", recallMacro)
    print("Micro recall: ", recallMicro)
    print("F1", f1score)
    print(confMat)


# Parse Arguments

if len(sys.argv) < 3:
    print("Please enter arguments: size and dataset")
model_size = sys.argv[1]
tf_dataset = sys.argv[2]

trainingData, testingData, trainingLabels, testingLabels, input_shape, num_classes = download_data(tf_dataset)
