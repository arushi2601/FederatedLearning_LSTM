import tensorflow as tf
import Fedrated_learning as fl
import csv

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize the image data
train_images = train_images / 255.0
print(len(train_images))
test_images = test_images / 255.0
print(len(test_images))

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

client_datasets = [client_dataset.batch(BATCH_SIZE).shuffle(SHUFFLE_BUFFER_SIZE).prefetch(1) for client_dataset in
                   client_datasets]
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(1)

print(tf.data.experimental.cardinality(test_dataset).numpy())

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

global_loss = []
global_accuracy = []
for i in range(300):
    for j in range(3):
        loss, accuracy = fl.local_model(client_train_list[j], client_test_list[j], 5, j)
        print(f"Client {j} model has loss of {loss} and accuracy of {accuracy}")

    model0 = tf.keras.models.load_model("client_model_0.h5")
    model1 = tf.keras.models.load_model("client_model_1.h5")
    model2 = tf.keras.models.load_model("client_model_2.h5")
    g_loss, g_accuracy = fl.global_model(model0, model1, model2, test_images, test_labels)
    global_loss.append(g_loss)
    global_accuracy.append(g_accuracy)
    print(f" Accuracy of global Model : {global_accuracy[i]} in comm round {i+1}.")

with open('GlobalData.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Global Loss', 'Global Accuracy'])
    for i in range(len(global_loss)):
        writer.writerow([global_loss[i], global_accuracy[i]])
