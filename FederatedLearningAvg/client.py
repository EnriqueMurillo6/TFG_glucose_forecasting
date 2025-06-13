import torch
from data import cargar_datos_paciente
from model import CustomTFT

class FlowerClient(flwr.client.NumPyClient):
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def get_parameters(self):
        return [param.detach().numpy() for param in self.model.parameters()]

    def set_parameters(self, parameters):
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_dataloader = self.dataset.to_dataloader(train=True, batch_size=BATCH_SIZE)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        
        for epoch in range(10):
            for batch in train_dataloader:
                y_pred = self.model(batch)
                loss = QuantileLoss(quantiles=[0.5])(y_pred, batch["Value"])
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        
        return self.get_parameters(), len(train_dataloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        val_dataloader = self.dataset.to_dataloader(train=False, batch_size=128)

        self.model.eval()
        validation_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                y_pred = self.model(batch)
                loss = QuantileLoss(quantiles=[0.5])(y_pred, batch["Value"])
                validation_loss += loss.item()

        avg_validation_loss = validation_loss / len(val_dataloader)
        return avg_validation_loss, len(val_dataloader.dataset), {}
