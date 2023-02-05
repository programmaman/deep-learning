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


def CreateTransformer(size, outputNeurons, shape, trainingData):
    if size == "tiny":
        transformer_layers = 4
    elif size == "small":
        transformer_layers = 6
    elif size == "base":
        transformer_layers = 8

    # do some data preprocessing
    data_augmentation = keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.Normalization(),
            tf.keras.layers.experimental.preprocessing.Resizing(image_size, image_size),
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(factor=0.02),
            tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=0.2, width_factor=0.2),
        ],
        name="data_augmentation",
    )
    # Compute the mean and the variance of the training data for normalization.
    data_augmentation.layers[0].adapt(trainingData)

    # set input layer/shape
    inputs = layers.Input(shape=shape)

    # Augment data
    augmented = data_augmentation(inputs)

    # Create patches
    patches = Patches(patch_size)(augmented)

    # Encode patches
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Final normalization/output
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)

    # add mlp to transformer
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)

    # pass features from mlp to final dense layer/classification
    logits = layers.Dense(outputNeurons)(features)

    # create model
    model = keras.Model(inputs=inputs, outputs=logits)

    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(),
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    return model


def create_cnn(size, numFilters, kernelSize, poolSize, numClasses, input_shape):
    if size == "tiny":
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(numFilters, (kernelSize, kernelSize), activation=activation,
                                   input_shape=input_shape),
            tf.keras.layers.MaxPooling2D(poolSize, poolSize),
            tf.keras.layers.Conv2D(numFilters * 2, (kernelSize, kernelSize), activation=activation),
            tf.keras.layers.MaxPooling2D(poolSize, poolSize),
            tf.keras.layers.Flatten(),  # flatten features
            tf.keras.layers.Dense(1024, activation=activation),
            tf.keras.layers.Dense(numClasses, outAct)
        ])

    elif size == "small":
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(numFilters, (kernelSize, kernelSize), activation=activation,
                                   input_shape=input_shape),
            tf.keras.layers.MaxPooling2D(poolSize, poolSize),
            tf.keras.layers.Conv2D(numFilters * 2, (kernelSize, kernelSize), activation=activation),
            tf.keras.layers.MaxPooling2D(poolSize, poolSize),
            tf.keras.layers.Conv2D(numFilters * 4, (kernelSize, kernelSize), activation=activation),
            tf.keras.layers.MaxPooling2D(poolSize, poolSize),
            tf.keras.layers.Flatten(),  # flatten features
            tf.keras.layers.Dense(1024, activation=activation),
            tf.keras.layers.Dense(2048, activation=activation),
            tf.keras.layers.Dense(numClasses, outAct)
        ])

    elif size == "base":
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(numFilters, (kernelSize, kernelSize), activation=activation,
                                   input_shape=input_shape),
            tf.keras.layers.MaxPooling2D(poolSize, poolSize),
            tf.keras.layers.Conv2D(numFilters * 2, (kernelSize, kernelSize), activation=activation),
            tf.keras.layers.MaxPooling2D(poolSize, poolSize),
            tf.keras.layers.Conv2D(numFilters * 4, (kernelSize, kernelSize), activation=activation),
            tf.keras.layers.Conv2D(numFilters * 8, (kernelSize, kernelSize), activation=activation),
            # tf.keras.layers.MaxPooling2D(poolSize, poolSize),
            tf.keras.layers.Flatten(),  # flatten features
            tf.keras.layers.Dense(1024, activation=activation),
            tf.keras.layers.Dense(2048, activation=activation),
            tf.keras.layers.Dense(4096, activation=activation),
            tf.keras.layers.Dense(numClasses, outAct)
        ])

    # configure the model (Not compile in the CS sense of the word)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        # set optimizer to Adam; for now know that optimizers help minimize loss (how to change weights)
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        # sparce categorical cross entropy (measure predicted dist vs. actual)
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(),
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )
    return model


def run_experiment(model, size):
    checkpoint_filepath = "./tmp/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=trainingData,
        y=trainingLabels,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[checkpoint_callback],
    )

    _, categorical, accuracy, top_5_accuracy = model.evaluate(testingData, testingLabels)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")
    print(print(f"Categorical Accuracy: {round(categorical * 100, 2)}%"))
    output_metrics(model, testingData, testingLabels)

    return history


def run_cnn_experiment(model, size):
    checkpoint = [tf.keras.callbacks.ModelCheckpoint(filepath="./" + size + '_CNN.h5',
                                                     monitor='sparse_categorical_accuracy',
                                                     verbose=1,
                                                     save_best_only=True,
                                                     save_weights=True,
                                                     mode='max'),
                  tf.keras.callbacks.EarlyStopping(monitor='sparse_categorical_accuracy',
                                                   min_delta=.01,
                                                   patience=5)]

    model.fit(trainingData, trainingLabels, epochs=num_epochs, callbacks=checkpoint)
    loaded_model = tf.keras.models.load_model("./" + size + '_CNN.h5')
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

# ViT Hyper-parameters
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 15
image_size = 72  # We'll resize input images to this size
patch_size = 6  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [projection_dim * 2, projection_dim]
transformer_layers = 8
mlp_head_units = [2048, 1024]

# CNN Hyper Parameters
activation = 'relu'
outAct = 'softmax'
numFilters = 32
kernelSize = 3
poolSize = 2

trainingData, testingData, trainingLabels, testingLabels, input_shape, num_classes = download_data(tf_dataset)

vit_classifier = CreateTransformer(model_size, num_classes, input_shape, trainingData)
history = run_experiment(vit_classifier, model_size)

cnn_model = create_cnn(model_size, numFilters, kernelSize, poolSize, num_classes, input_shape)
run_cnn_experiment(cnn_model, model_size)
