from typing import List, Tuple
from flwr.common import Metrics
from flwr.server.strategy import FedAvg
from flwr.server.app import ServerApp, ServerAppComponents
from flwr.server.config import ServerConfig
from flwr.server.context import Context

def weighted_rmse(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    rmses = [num_examples * m["rmse"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"rmse": sum(rmses) / sum(examples)}

def server_fn(context: Context) -> ServerAppComponents:
    """Configura el servidor para federated learning con FedAvg.

    Args:
        context: Contexto que provee configuración para el servidor.

    Returns:
        ServerAppComponents con estrategia y configuración para el servidor.
    """

    strategy = FedAvg(
        fraction_fit=0.75,            # Usar el 75% de clientes para entrenamiento en cada ronda
        fraction_evaluate=0.75,       # Usar el 75% para evaluación
        min_fit_clients=15,           # Mínimo 15 clientes entrenando cada ronda
        min_evaluate_clients=15,      # Mínimo 15 clientes evaluando cada ronda
        min_available_clients=15,     # Esperar hasta tener 15 clientes disponibles
        evaluate_metrics_aggregation_fn=weighted_rmse,
    )

    config = ServerConfig(num_rounds=5)  # Configurar para 5 rondas de entrenamiento

    return ServerAppComponents(strategy=strategy, config=config)

server = ServerApp(server_fn=server_fn)
