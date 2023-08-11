import tensorflow as tf
import numpy as np

def local_model(train_images, test_images, epochs, i):
    # Create a simple neural network model
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(train_images, epochs=epochs)
    loss, accuracy = model.evaluate(test_images)
    filename = f"client_model_{i}.h5"
    model.save(filename)
    return loss, accuracy


def global_model(model1, model2, model3, test_images, test_labels):
    weights = []
    for model in [model1, model2, model3]:
        weights.append(model.get_weights())

    avg_weights = []
    for i in range(len(weights[0])):
        layer_weights = np.array([weights[j][i] for j in range(len(weights))])
        avg_layer_weights = np.mean(layer_weights, axis=0)
        avg_weights.append(avg_layer_weights)

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.set_weights(avg_weights)

    # Evaluate the final model on the test dataset
    loss, accuracy = model.evaluate(test_images, test_labels)
    return loss, accuracy
