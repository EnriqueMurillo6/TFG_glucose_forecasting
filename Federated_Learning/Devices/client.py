import warnings
warnings.filterwarnings("ignore")
import lightning.pytorch as pl
from flwr.common import Context
from datasets     import load_datasets
from models       import load_model
import torch
from flwr.client import NumPyClient, ClientApp
import torch
import torch.nn.functional as F
import pandas as pd
from flwr.common import Parameters
from flwr.common import ndarrays_to_parameters
from datasets import BATCH_SIZE

df = pd.read_parquet("../df_15.parquet")

DEFAULT_LR        = 0.001
DEFAULT_EPOCHS    = 5
DEFAULT_BATCHSIZE = BATCH_SIZE

def state_dict_to_list(state_dict):
    """Convierte un state_dict en lista de ndarrays (orden fija)."""
    return [v.detach().cpu().numpy() for v in state_dict.values()]


def list_to_state_dict(param_list, ref_state_dict, device="cpu"):
    """Reconstruye un state_dict con las claves de ref_state_dict."""
    keys = list(ref_state_dict.keys())
    return {k: torch.tensor(v, device=device) for k, v in zip(keys, param_list)}

class FlowerClient(NumPyClient):
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        epochs: int,
        learning_rate: float,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model         = model.to(self.device)
        self.train_loader  = train_loader
        self.val_loader    = val_loader
        self.epochs        = epochs
        self.learning_rate = learning_rate

    def _set_weights(self, parameters):
        sd = list_to_state_dict(parameters, self.model.state_dict(), device=self.device)
        self.model.load_state_dict(sd, strict=True)
    def _state_dict_cpu(self):
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

    def get_parameters(self, config=None):
        return state_dict_to_list(self._state_dict_cpu())
    def fit(self, parameters, config=None):
        self._set_weights(parameters)

        trainer = pl.Trainer(
            max_epochs=self.epochs,
            accelerator="gpu",
            devices=[0],
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
        )
        trainer.fit(self.model, train_dataloaders=self.train_loader, val_dataloaders=self.val_loader)


        return state_dict_to_list(self.model.state_dict()), len(self.train_loader.dataset), {}
    def evaluate(self, parameters, config=None):
        self._set_weights(parameters)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=[0],
            logger=False,
            enable_checkpointing=False,
        )
        test_dataloader = self.val_loader


        raw_predictions = self.model.predict(test_dataloader, mode="prediction", return_y=True)
        y_pred = raw_predictions.output
        y_true = raw_predictions.y[0]

        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()

        from sklearn.metrics import mean_absolute_error, root_mean_squared_error
        rmse = float(root_mean_squared_error(y_true, y_pred))
        mae = float(mean_absolute_error(y_true, y_pred))

        return rmse, len(test_dataloader.dataset), {"rmse": rmse, "mae": mae}

def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    node_id      = context.node_id
    print(f"[Client {node_id}] → partición {partition_id}")
  
    run_cfg       = context.run_config or {}
    learning_rate = run_cfg.get("learning-rate", DEFAULT_LR)
    epochs        = run_cfg.get("local-epochs",  DEFAULT_EPOCHS)
    batch_size    = run_cfg.get("batch-size",    DEFAULT_BATCHSIZE)
  
    train_loader, val_loader, training_ds = load_datasets(
        df,
        partition_id=partition_id
    )
    model = load_model(learning_rate=learning_rate, training=training_ds)

    return FlowerClient(
        model=model,
        learning_rate=learning_rate,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
    ).to_client()

app = ClientApp(client_fn=client_fn)
