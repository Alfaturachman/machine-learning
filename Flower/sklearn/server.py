import flwr as fl

# Menjalankan server Flower
if __name__ == "__main__":
    # Mengonfigurasi server untuk 3 putaran federated learning
    fl.server.start_server(
        server_address="localhost:8080",  # Menetapkan alamat server
        config=fl.server.ServerConfig(num_rounds=3)  # Menggunakan ServerConfig
    )
