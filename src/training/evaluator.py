import torch


class Evaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def evaluate(self, data_loader):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                out, _ = self.model(data.x, data.edge_index, data.batch)
                pred = out.argmax(dim=1)
                correct += int((pred == data.y).sum())
                total += data.num_graphs

        acc = correct / total if total > 0 else 0.0
        return acc
