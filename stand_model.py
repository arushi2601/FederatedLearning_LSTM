import tensorflow as tf


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize the image data
train_images = train_images / 255.0
print(len(train_images))
test_images = test_images / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)


test_images = test_images.reshape((-1, 28, 28))
loss, accuracy = model.evaluate(test_images, test_labels)
print("Test loss:", loss)
print("Test accuracy:", accuracy)
