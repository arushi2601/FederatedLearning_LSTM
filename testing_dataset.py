import tensorflow as tf
import csv

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize the image data
train_images = train_images / 255.0
print(len(train_images))
test_images = test_images / 255.0


# Create a dataset from the training data
dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))

# Shuffle the dataset
dataset = dataset.shuffle(buffer_size=len(train_images))

# Split the dataset into three parts
client_datasets = []
for i in range(3):
    client_dataset = dataset.skip(i * len(train_images) // 3).take(len(train_images) // 3)
    client_datasets.append(client_dataset)

# Create a test dataset
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

# Batch and prefetch the datasets
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

client_datasets = [client_dataset.batch(BATCH_SIZE).shuffle(SHUFFLE_BUFFER_SIZE).prefetch(1) for client_dataset in client_datasets]
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(1)

# Print the size of each client's dataset
for i, client_dataset in enumerate(client_datasets):
    print("Client", i, "dataset size:", tf.data.experimental.cardinality(client_dataset).numpy())

# Split the first client dataset into train and test sets
client_train_list = []
client_test_list = []

for i, client_dataset in enumerate(client_datasets):
    client_train = client_dataset.take(int(0.8 * tf.data.experimental.cardinality(client_dataset).numpy()))
    client_test = client_dataset.skip(int(0.8 * tf.data.experimental.cardinality(client_dataset).numpy()))
    client_train_list.append(client_train)
    client_test_list.append(client_test)

print(client_train_list)



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

# Train the model on the client 0 training dataset

for i in range(3):
    model.fit(client_train_list[i], epochs=5)
    # Evaluate the model on the client 0 test dataset
    loss, accuracy = model.evaluate(client_test_list[i])
    print("Test loss:", loss)
    print("Test accuracy:", accuracy)
    filename = f"client_model_{i}.h5"
    model.save(filename)



