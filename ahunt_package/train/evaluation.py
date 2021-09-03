import torch
from utils import device


def evaluate_model(model, test_loader):
    actuals, predictions = torch.tensor([]), torch.tensor([])
    model.eval()
    model.to(device)

    with torch.no_grad():
        for i, (data, target, _) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            preds = output.argmax(dim=1).to("cpu")
            predictions = torch.cat((predictions, preds), dim=0)
            actuals = torch.cat((actuals, target.to("cpu")), dim=0)
    return actuals, predictions