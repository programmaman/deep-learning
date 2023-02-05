"""
This is code is modfied from https://keras.io/examples/vision/image_classification_with_vision_transformer/.
It will not run as is. There are variables that are not defined and imports are missing.
If you use this code, you will need to define/import.
This code is so you can have a skeleton/baseline to implement a vision transformer. It uses the functional API in Keras.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import sys
from sklearn.metrics import precision_score, recall_score, confusion_matrix


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

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


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def download_data(name):
    if name == 'fashion_mnist':
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (trainingData, trainingLabels), (testingData, testingLabels) = fashion_mnist.load_data()
        trainingData, testingData = trainingData / 255.0, testingData / 255.0
        input_shape = (28, 28, 1)
        numClasses = 10
        print(f"trainingData shape: {trainingData.shape} - trainingLabels shape: {trainingLabels.shape}")
        print(f"testingData shape: {testingData.shape} - testingLabels shape: {testingLabels.shape}")
        return trainingData, testingData, trainingLabels, testingLabels, input_shape, numClasses

    if name == 'cifar10':
        cifar10 = tf.keras.datasets.cifar10
        (trainingData, trainingLabels), (testingData, testingLabels) = cifar10.load_data()
        trainingData, testingData = trainingData / 255.0, testingData / 255.0
        input_shape = (32, 32, 3)
        numClasses = 10
        print(f"trainingData shape: {trainingData.shape} - trainingLabels shape: {trainingLabels.shape}")
        print(f"testingData shape: {testingData.shape} - testingLabels shape: {testingLabels.shape}")
        return trainingData, testingData, trainingLabels, testingLabels, input_shape, numClasses

    if name == 'cifar100':
        cifar100 = tf.keras.datasets.cifar100
        (trainingData, trainingLabels), (testingData, testingLabels) = cifar100.load_data()
        trainingData, testingData = trainingData / 255.0, testingData / 255.0
        input_shape = (32, 32, 3)
        numClasses = 100
        print(f"trainingData shape: {trainingData.shape} - trainingLabels shape: {trainingLabels.shape}")
        print(f"testingData shape: {testingData.shape} - testingLabels shape: {testingLabels.shape}")
        return trainingData, testingData, trainingLabels, testingLabels, input_shape, numClasses


def display_sample_patch():
    plt.figure(figsize=(4, 4))
    image = trainingData[np.random.choice(range(trainingData.shape[0]))]
    plt.imshow(image.astype("uint8"))
    plt.axis("off")

    resized_image = tf.image.resize(
        tf.convert_to_tensor([image]), size=(image_size, image_size)
    )
    patches = Patches(patch_size)(resized_image)
    print(f"Image size: {image_size} X {image_size}")
    print(f"Patch size: {patch_size} X {patch_size}")
    print(f"Patches per image: {patches.shape[1]}")
    print(f"Elements per patch: {patches.shape[-1]}")

    n = int(np.sqrt(patches.shape[1]))
    plt.figure(figsize=(4, 4))
    for i, patch in enumerate(patches[0]):
        ax = plt.subplot(n, n, i + 1)
        patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
        plt.imshow(patch_img.numpy().astype("uint8"))
        plt.axis("off")


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


# VIT Hyper Parameters
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 2
image_size = 48  # We'll resize input images to this size
patch_size = 3  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [projection_dim * 2, projection_dim, ]
transformer_layers = 8
mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier




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
            tf.keras.layers.MaxPooling2D(poolSize, poolSize),
            tf.keras.layers.Flatten(),  # flatten features
            tf.keras.layers.Dense(4096, activation=activation),
            tf.keras.layers.Dense(numClasses, outAct)
        ])

    # configure the model (Not compile in the CS sense of the word)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        # set optimizer to Adam; for now know that optimizers help minimize loss (how to change weights)
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        # sparce categorical cross entropy (measure predicted dist vs. actual)
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(),
                 keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                 keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"), ],
    )
    return model


def create_vit(size, outputNeurons, input_shape, trainingData):
    if size == "tiny":
        transformer_layers = 4
    elif size == "small":
        transformer_layers = 8
    elif size == "base":
        transformer_layers = 12

    # do some data preprocessing
    data_augmentation = keras.Sequential(
        [
            layers.Normalization(),
            layers.Resizing(image_size, image_size),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.02),
            layers.RandomZoom(height_factor=0.2, width_factor=0.2),
        ],
        name="data_augmentation",
    )

    # Compute the mean and the variance of the training data for normalization.
    data_augmentation.layers[0].adapt(trainingData)

    # set input layer/shape
    inputs = layers.Input(shape=input_shape)

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
    optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)

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
    saved_model = tf.keras.models.load_model("./" + size + '_CNN.h5')
    output_metrics(saved_model, trainingData, trainingLabels)


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

    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(testingData, testingLabels)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")
    model.save('./' + size + '_vision_transformer.h5')


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


# Parse Arguments
if len(sys.argv) < 3:
    print("Please enter arguments: size and dataset")
model_size = sys.argv[1]
tf_dataset = sys.argv[2]

# Train
trainingData, testingData, trainingLabels, testingLabels, input_shape, numClasses = download_data(tf_dataset)

vit_model = create_vit(model_size, numClasses, input_shape, trainingData)
run_experiment(vit_model, model_size)
cnn_model = create_cnn(model_size, numFilters, kernelSize, poolSize, numClasses, input_shape)
run_cnn_experiment(cnn_model, model_size)
