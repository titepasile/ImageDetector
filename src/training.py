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
        train_epoch(model, training_split, criterion, optimizer, device)
        if i % scheduler_interv == 0 and i > 0:
            scheduler.step()


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()
