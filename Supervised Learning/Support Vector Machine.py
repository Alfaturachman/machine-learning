# Import necessary libraries
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset iris (dataset bawaan scikit-learn)
iris = datasets.load_iris()
X = iris.data[:, :2]  # Menggunakan dua fitur pertama untuk memudahkan visualisasi
y = iris.target

# Membagi data menjadi data training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Membuat model SVM dengan kernel linear
svm_model = SVC(kernel='linear')

# Melatih model dengan data training
svm_model.fit(X_train, y_train)

# Prediksi dengan data testing
y_pred = svm_model.predict(X_test)

# Menampilkan akurasi
print(f"Akurasi: {accuracy_score(y_test, y_pred)}")

# Visualisasi dataset dan hyperplane (untuk 2 fitur saja)
def plot_svm_boundary(model, X, y):
    # Buat grid untuk plot
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Prediksi seluruh titik dalam grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot contour dan data
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.xlabel('Fitur 1')
    plt.ylabel('Fitur 2')
    plt.title('Garis Pemisah SVM (Hyperplane)')
    plt.show()

# Plot hyperplane dan data
plot_svm_boundary(svm_model, X_test, y_test)
