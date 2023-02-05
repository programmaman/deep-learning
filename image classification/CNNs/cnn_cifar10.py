# Code modified from https://www.tensorflow.org/
# Code modified from Dr. S. Canavan

import tensorflow as tf



# download, split, and normalize dataset
def download_shape_Data():
    cifar10 = tf.keras.datasets.cifar10
    (trainingData, trainingLabels), (testingData, testingLabels) = cifar10.load_data()
    trainingData, testingData = trainingData / 255.0, testingData / 255.0
    return trainingData, testingData, trainingLabels, testingLabels


# create and configure network architecture
def CreateModel(act, outAct, numFilters, kernelSize, poolSize, denseNeuronSize, numClasses):
    # make a sequential model (stack of layers in order)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(numFilters, (kernelSize, kernelSize), activation=act, input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D(poolSize, poolSize),
        tf.keras.layers.Conv2D(numFilters * 2, (kernelSize, kernelSize), activation=act),
        tf.keras.layers.MaxPooling2D(poolSize, poolSize),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(numFilters * 4, (kernelSize, kernelSize), activation=act),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(numFilters * 8, (kernelSize, kernelSize), activation=act),
        tf.keras.layers.Flatten(),  # flatten features
        tf.keras.layers.Dense(denseNeuronSize, activation=act),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(denseNeuronSize, activation=act, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.001, l2=0.001)),
        tf.keras.layers.Dense(numClasses, activation=outAct)  # 1 of 10 classes
    ])

    # configure the model (Not compile in the CS sense of the word)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        # set optimizer to Adam; for now know that optimizers help minimize loss (how to change weights)
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        # sparce categorical cross entropy (measure predicted dist vs. actual)
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],  # how often do predictions match labels
    )
    return model


def main():
    act = 'relu'
    outAct = 'softmax'
    numFilters = 32
    kernelSize = 3
    poolSize = 2
    denseNeuronSize = 2048
    numClasses = 10
    batch_size = 128
    epochs = 25

    # get data
    trainingData, testingData, trainingLabels, testingLabels = download_shape_Data()

    # create network
    model = CreateModel(act, outAct, numFilters, kernelSize, poolSize, denseNeuronSize, numClasses)

    # print out the summary of the model
    model.summary()

    # train model (no validation data)
    model.fit(trainingData, trainingLabels, batch_size, epochs)

    # evaluate function on all testing data
    model.evaluate(testingData, testingLabels)



if __name__ == "__main__":
    main()
