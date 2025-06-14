import torch
import torch.nn as nn

def train(net, trainloader, epochs: int, optimizer=None, verbose=False):
    net.train()
    loss_fn = net.loss if hasattr(net, 'loss') else nn.MSELoss()

    if optimizer is None:
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in trainloader:
            optimizer.zero_grad()
            x = batch[0].to(DEVICE)
            y = batch[0]['decoder_target'].to(DEVICE)

            output = net(x)

            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(trainloader):.4f}")


def test(net, testloader):
    net.eval()
    loss_fn = net.loss if hasattr(net, 'loss') else nn.MSELoss()
    total_loss = 0.0

    with torch.no_grad():
        for batch in testloader:
            x = batch[0].to(DEVICE)
            y = batch[0]['decoder_target'].to(DEVICE)

            output = net(x)
            loss = loss_fn(output, y)
            total_loss += loss.item()

    avg_loss = total_loss / len(testloader)
    return avg_loss
