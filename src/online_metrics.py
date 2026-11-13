import torch


class OnlineMetrics:
    def __init__(self, device):
        self.device = torch.device(device)
        self.reset()

    def reset(self):
        dtype = torch.float32 if self.device.type == "mps" else torch.float64

        # [weighted_loss, batch_loss_sum, batch_loss_sq_sum, correct]
        self.stats = torch.zeros(4, device=self.device, dtype=dtype)

        self.count = 0
        self.total_samples = 0

    @torch.no_grad()
    def update(self, loss, logits, targets):
        batch_size = targets.size(0)

        batch_loss = loss.detach().to(self.stats.dtype)
        correct = (logits.argmax(dim=1) == targets).sum()

        self.stats.add_(
            torch.stack(
                (
                    batch_loss * batch_size,
                    batch_loss,
                    batch_loss.square(),
                    correct,
                )
            )
        )

        self.count += 1
        self.total_samples += batch_size

    def get_dict(self):
        if self.count == 0:
            return None

        # One CPU-GPU synchronization per epoch
        total_loss, loss_sum, loss_sq_sum, correct = self.stats.cpu().tolist()

        variance = None

        if self.count > 1:
            variance = max(
                0.0,
                (loss_sq_sum - loss_sum**2 / self.count) / (self.count - 1),
            )

        return {
            "train_total_loss": total_loss,
            "train_mean_loss": total_loss / self.total_samples,
            "train_accuracy": correct / self.total_samples,
            "train_variance": variance,
        }
