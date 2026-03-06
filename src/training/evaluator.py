import torch


class Evaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def evaluate(self, loader):
        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)

                # ✅ v2 写法：直接传 data
                logits = self.model(data)

                pred = logits.argmax(dim=1)
                correct += int((pred == data.y).sum())
                total += data.num_graphs

        acc = correct / total
        return acc
