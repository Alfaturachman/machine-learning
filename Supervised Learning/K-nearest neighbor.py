# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Contoh dataset (data buatan)
# X adalah fitur (panjang dan lebar), y adalah label (0 dan 1)
X = np.array([[1, 1], [2, 2], [3, 3], [6, 6], [7, 7], [8, 8]])
y = np.array([0, 0, 0, 1, 1, 1])

# Split data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Membuat model KNN dengan K=3
knn = KNeighborsClassifier(n_neighbors=3)

# Melatih model
knn.fit(X_train, y_train)

# Prediksi dengan data testing
y_pred = knn.predict(X_test)

# Menampilkan hasil akurasi
print(f"Akurasi: {accuracy_score(y_test, y_pred)}")

# Visualisasi dataset dan hasil klasifikasi
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', label='Data Asli')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, edgecolor='k', label='Prediksi')
plt.title('Klasifikasi dengan KNN')
plt.xlabel('Fitur 1')
plt.ylabel('Fitur 2')
plt.legend()
plt.show()
