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
from keras import layers
import tensorflow_addons as tfa
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score


class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def get_config(self):
        config = super(Patches, self).get_config()
        config.update({"patch_size": self.patch_size})
        return config

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def get_config(self):
        config = super(PatchEncoder, self).get_config()
        config.update({"num_patches": self.num_patches, "projection_dim": self.projection_dim})
        return config

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def download_data(name):
    if name == 'fashion_mnist':
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        shape = (28, 28, 1)
        number_of_classes = 10
        return x_train, x_test, y_train, y_test, shape, number_of_classes
    if name == 'cifar10':
        cifar10 = tf.keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        shape = (32, 32, 3)
        number_of_classes = 10
        return x_train, x_test, y_train, y_test, shape, number_of_classes
    if name == 'cifar100':
        cifar100 = tf.keras.datasets.cifar100
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        shape = (32, 32, 3)
        number_of_classes = 100
        return x_train, x_test, y_train, y_test, shape, number_of_classes


def test_cnn(size):
    loaded_model = tf.keras.models.load_model("./" + size + '_CNN.h5')
    _, categorical, accuracy, top_5_accuracy = loaded_model.evaluate(testingData, testingLabels)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")
    print(print(f"Categorical Accuracy: {round(categorical * 100, 2)}%"))
    output_metrics(loaded_model, testingData, testingLabels)


def test_vit(size):
    loaded_model = tf.keras.models.load_model("./" + size + '_vision_transformer.h5',
                                              custom_objects={'Patches': Patches, 'PatchEncoder':PatchEncoder})
    _, categorical, accuracy, top_5_accuracy = loaded_model.evaluate(testingData, testingLabels)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")
    print(print(f"Categorical Accuracy: {round(categorical * 100, 2)}%"))
    output_metrics(loaded_model, testingData, testingLabels)


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
    f1score_micro = f1_score(labels, predict, average='macro')
    print("Macro precision: ", precisionMacro)
    print("Micro precision: ", precisionMicro)
    print("Macro recall: ", recallMacro)
    print("Micro recall: ", recallMicro)
    print("F1 Macro", f1score)
    print("F1 Micro", f1score_micro)
    print(confMat)


# Parse Arguments

if len(sys.argv) < 3:
    print("Please enter arguments: size and dataset")
model_size = sys.argv[1]
tf_dataset = sys.argv[2]

trainingData, testingData, trainingLabels, testingLabels, input_shape, num_classes = download_data(tf_dataset)
print("Testing ViT:")
test_vit(model_size)
print("Testing CNN:")
test_cnn(model_size)
