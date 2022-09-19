import torch
import tqdm


def train_step(model, data_loader, loss_fn, device, optimizer):
    model.train()
    loss, acc = 0, 0
    for batch, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)
        logits = model(x)

        model.zero_grad()
        l = loss_fn(logits, y)
        loss += l.item()

        l.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        acc += (y_pred_class == y).sum().item() / len(logits)

    loss /= len(data_loader)
    acc /= len(data_loader)
    return loss, acc


def eval(model, data_loader, loss_fn, device):
    model.eval()
    loss, acc = 0, 0
    with torch.inference_mode():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)

            l = loss_fn(logits, y)
            loss += l.item()

            y_pred_class = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            acc += (y_pred_class == y).sum().item() / len(logits)

    loss /= len(data_loader)
    acc /= len(data_loader)
    return loss, acc


def train(model, train_loader, test_loader, loss_fn, optimizer, logger=None):
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(logger.config.epochs):
        train_loss, train_acc = train_step(
            model=model,
            data_loader=train_loader,
            loss_fn=loss_fn,
            device=logger.config.device,
            optimizer=optimizer,
        )
        test_loss, test_acc = eval(
            model=model,
            data_loader=test_loader,
            loss_fn=loss_fn,
            device=logger.config.device,
        )

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        if logger:
            logger.log(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                }
            )

    # Return the filled results at the end of the epochs
    return results
