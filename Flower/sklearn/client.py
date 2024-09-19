import flwr as fl
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset lokal
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Membuat client Flower
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self):
        # Mengambil parameter dari model tanpa argumen 'config'
        return model.get_params()

    def set_parameters(self, parameters):
        # Menetapkan parameter ke model
        model.set_params(**parameters)

    def fit(self, parameters, config):
        # Melakukan pelatihan lokal
        self.set_parameters(parameters)
        model.fit(X_train, y_train)
        return self.get_parameters(), len(X_train), {}

    def evaluate(self, parameters, config):
        # Mengevaluasi model
        self.set_parameters(parameters)
        loss = model.score(X_test, y_test)
        return loss, len(X_test), {}

# Inisialisasi model
model = LogisticRegression()

# Memulai client Flower
if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="localhost:8080", client=FlowerClient())
