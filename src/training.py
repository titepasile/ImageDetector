from .online_metrics import OnlineMetrics
import mlflow


def fit(
    model,
    training_split,
    val_split,
    criterion,
    optimizer,
    scheduler,
    scheduler_interv,
    n_epochs,
    device,
):
    for i in range(n_epochs):
        metrics = train_epoch(model, training_split, criterion, optimizer, device)
        mlflow.log_metrics(metrics, step=i)
        scheduler.step()


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    metrics = OnlineMetrics()

    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()
        metrics.update(loss, out, y)

    return metrics.get_dict()
