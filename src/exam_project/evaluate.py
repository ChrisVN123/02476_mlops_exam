# src/exam_project/evaluate.py

import torch


def evaluate_model(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()  # Or pass this from main.py if needed

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, torch.argmax(y_batch, dim=1))
            test_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == torch.argmax(y_batch, dim=1)).sum().item()
            total += y_batch.size(0)

    test_loss /= len(test_loader)
    accuracy = correct / total
    return test_loss, accuracy
