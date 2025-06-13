import flwr
strategy = flwr.server.strategy.FedAvg(
    fraction_fit=0.1,
    min_fit_clients=2,
    min_available_clients=2
)

flwr.server.start_server(
    server_address="127.0.0.1:8080",
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    client_manager=fl.server.SimpleClientManager(),
    strategy=strategy
)
