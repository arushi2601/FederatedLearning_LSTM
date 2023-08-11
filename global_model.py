import tensorflow as tf
import numpy as np
import csv

(_, _), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize the image data
test_images = test_images / 255.0


# # Reshape the test images to match the input shape of the model


# Create the final model by averaging the weights of the three models
model1 = tf.keras.models.load_model("client_model_0.h5")
model2 = tf.keras.models.load_model("client_model_1.h5")
model3 = tf.keras.models.load_model("client_model_2.h5")

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
print("Test loss:", loss)
print("Test accuracy:", accuracy)

# Make predictions on a few test images
predictions = model.predict(test_images[:10])
print(predictions)
