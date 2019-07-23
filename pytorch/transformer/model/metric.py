import torch
from tqdm import tqdm


def evaluate(model, data_loader, metrics, device):
    if model.training:
        model.eval()

    summary = {metric: 0 for metric in metrics}

    for step, mb in tqdm(enumerate(data_loader), desc='steps', total=len(data_loader)):
        sent1, sent2, label = map(lambda elm: elm.to(device), mb)

        with torch.no_grad():
            logits = model(sent1, sent2)

            for metric in metrics:
                summary[metric] += metrics[metric](logits, label).item()
    else:
        for metric in metrics:
            summary[metric] /= (step + 1)

    return summary


def acc(yhat, y):
    with torch.no_grad():
        yhat = yhat.max(dim=1)[1]
        acc = (yhat == y).float().mean()
    return acc