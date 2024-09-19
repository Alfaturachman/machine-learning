import flwr as fl

# Define custom aggregation or server logic if needed
def aggregate_fn(results):
    # Custom aggregation logic here, if needed
    return fl.server.strategy.FedAvg.aggregate(results)

# Start the server with additional configurations
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3, strategy=fl.server.strategy.FedAvg(aggregate_fn=aggregate_fn))
)
