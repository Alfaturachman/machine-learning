import tensorflow as tf
import flwr as fl
import numpy as np

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = y_train.flatten(), y_test.flatten()

# Define the model within the FlowerClient class
class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = tf.keras.applications.MobileNetV2(input_shape=(32, 32, 3), classes=10, weights=None)
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        # Optionally: Implement data augmentation and other preprocessing steps

    def get_parameters(self, config):
        return self.model.get_weights()
    
    def fit(self, parameters, config):
        self.model.set_weights(parameters)  # Set the weights from the server

        # Optionally: Implement custom training loops or additional callbacks
        self.model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=1)
        
        updated_parameters = self.model.get_weights()  # Get updated weights
        return updated_parameters, len(x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)  # Set the weights from the server
        
        # Optionally: Implement additional metrics or evaluation steps
        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=1)
        return loss, len(x_test), {"accuracy": accuracy}  # Return evaluation results

# Create and start the Flower client
client = FlowerClient()
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
