import torch

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc

def compute_F1(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))

    TP = rounded_preds.eq(1).mul(y.eq(1)).float().sum()
    TN = rounded_preds.eq(0).mul(y.eq(0)).float().sum()
    FN = rounded_preds.eq(0).mul(y.eq(1)).float().sum()
    FP = rounded_preds.eq(1).mul(y.eq(0)).float().sum()

    return TP, TN, FN, FP

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def test_evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    TP = 0
    TN = 0
    FN = 0
    FP = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

            _TP, _TN, _FN, _FP = compute_F1(predictions, batch.label)
            TP += _TP
            TN += _TN
            FN += _FN
            FP += _FP

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * recall * precision / (recall + precision)

    return epoch_loss / len(iterator), epoch_acc / len(iterator), precision, recall, F1

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)