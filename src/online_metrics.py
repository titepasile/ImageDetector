class OnlineMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_loss = 0
        self.count = 0
        self.total_correct = 0
        self.total_samples = 0
        self.mean = 0
        self.moment = 0

    def update(self, loss, logits, targets):
        loss = loss.detach().item()
        batch_size = targets.size(0)

        self.count += 1
        delta1 = loss - self.mean
        self.mean += delta1 / self.count

        self.total_loss += loss * batch_size
        self.total_samples += batch_size
        delta2 = loss - self.mean

        self.moment += delta1 * delta2

        predictions = logits.argmax(dim=1)
        self.total_correct += (predictions == targets).sum().item()

    def get_dict(self):
        if self.count == 0:
            return None
        var = self.moment / (self.count - 1) if self.count > 1 else None
        metrics = {
            "train_total_loss": self.total_loss,
            "train_mean_loss": self.total_loss / self.total_samples,
            "train_accuracy": self.total_correct / self.total_samples,
            "train_variance": var,
        }
        return metrics
