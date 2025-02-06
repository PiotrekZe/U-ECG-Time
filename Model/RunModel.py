import numpy as np
from collections import Counter
import torch
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score


def categorize_data(targets, preds, peaks):
    targets = np.array(targets.cpu())
    preds = np.array(preds.cpu())
    peaks = np.array(peaks.cpu())
    indices_of_ones = [np.where(row == 1)[0] for row in peaks]
    categorized_preds_most, categorized_targets = [], []

    for i in range(len(peaks)):
        tmp_preds_most, tmp_targets = [], []
        for j in range(len(indices_of_ones[i])):
            if j == 0 and indices_of_ones[i][j] != 0:
                counter = Counter(preds[i][0 : indices_of_ones[i][0]])
                most_common_number, count = counter.most_common(1)[0]
                tmp_preds_most.append(most_common_number)

                counter = Counter(targets[i][0 : indices_of_ones[i][0]])
                counter_target, count = counter.most_common(1)[0]
            elif j == len(indices_of_ones[i]) - 1:
                counter = Counter(preds[i][indices_of_ones[i][-1] : len(peaks[i])])
                most_common_number, count = counter.most_common(1)[0]
                tmp_preds_most.append(most_common_number)

                counter = Counter(targets[i][indices_of_ones[i][-1] : len(peaks[i])])
                counter_target, count = counter.most_common(1)[0]
            else:
                counter = Counter(
                    preds[i][indices_of_ones[i][j] : indices_of_ones[i][j + 1]]
                )
                most_common_number, count = counter.most_common(1)[0]
                tmp_preds_most.append(most_common_number)

                counter = Counter(
                    targets[i][indices_of_ones[i][j] : indices_of_ones[i][j + 1]]
                )
                counter_target, count = counter.most_common(1)[0]
            tmp_targets.append(counter_target)

        categorized_preds_most.append(tmp_preds_most)
        categorized_targets.append(tmp_targets)
    return categorized_preds_most, categorized_targets


def calculate_metrices(preds, targets, num_classes, end_idxs=None):
    accuracy, precision, recall, f1 = [], [], [], []
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for i in range(len(targets)):
        if end_idxs is None:
            tmp_true = targets[i]
            tmp_pred = preds[i]
        else:
            if end_idxs[i].item() == 0:
                continue
            else:
                tmp_true = targets[i][:-1]
                tmp_pred = preds[i][:-1]

        accuracy.append(accuracy_score(tmp_true, tmp_pred))
        precision.append(
            precision_score(tmp_true, tmp_pred, zero_division=0, average="weighted")
        )
        recall.append(
            recall_score(tmp_true, tmp_pred, zero_division=0, average="weighted")
        )
        f1.append(f1_score(tmp_true, tmp_pred, zero_division=0, average="weighted"))
        cm += confusion_matrix(tmp_true, tmp_pred, labels=np.arange(num_classes))

    return (
        np.nanmean(accuracy),
        np.nanmean(precision),
        np.nanmean(recall),
        np.nanmean(f1),
        cm,
    )


def test_model(model, criterion, data_loader, device, num_classes, num_channels=2):
    model.eval()

    iou_score = []
    acc_tab, prec_tab, rec_tab, f1_tab = [], [], [], []
    cm_tab = np.zeros((num_classes, num_classes), dtype=int)

    running_loss = 0

    with torch.no_grad():
        for inputs, targets, peaks, end_idxs in data_loader:
            inputs = inputs.to(torch.float32).to(device)
            targets = targets.to(device)
            peaks = peaks.to(device)
            end_idxs = end_idxs.to(device)

            batch_outputs = model(inputs)
            outputs = batch_outputs.permute(0, 2, 1).contiguous()
            outputs = outputs.view(-1, num_classes)
            target = targets.view(-1)

            loss = criterion(outputs, target)
            running_loss += loss.item()

            preds = torch.argmax(batch_outputs, dim=1)

            # categorized normal
            categorized_preds, categorized_targets = categorize_data(
                targets, preds, peaks
            )
            acc, prec, rec, f1, cm = calculate_metrices(
                preds=categorized_preds,
                targets=categorized_targets,
                num_classes=num_classes,
                end_idxs=end_idxs,
            )
            acc_tab.append(acc)
            prec_tab.append(prec)
            rec_tab.append(rec)
            f1_tab.append(f1)
            cm_tab += cm

    print("Testing!")
    print(
        f"Loss value: {running_loss/len(data_loader)}, Accuracy: {np.nanmean(acc_tab)}"
    )

    return_list = [
        running_loss / len(data_loader),
        np.nanmean(acc_tab),
        np.nanmean(prec_tab),
        np.nanmean(rec_tab),
        np.nanmean(f1_tab),
        cm_tab,
    ]

    return return_list


def train_model(
    model, criterion, optimizer, data_loader, device, num_classes, num_channels=2
):
    model.train()

    iou_score = []
    acc_tab, prec_tab, rec_tab, f1_tab = [], [], [], []
    cm_tab = np.zeros((num_classes, num_classes), dtype=int)

    running_loss = 0

    for inputs, targets, peaks, end_idxs in data_loader:
        inputs = inputs.to(torch.float32).to(device)
        targets = targets.to(device)
        peaks = peaks.to(device)
        end_idxs = end_idxs.to(device)

        optimizer.zero_grad()
        batch_outputs = model(inputs)

        outputs = batch_outputs.permute(0, 2, 1).contiguous()
        outputs = outputs.view(-1, num_classes)
        target = targets.view(-1)

        loss = criterion(outputs, target)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        preds = torch.argmax(batch_outputs, dim=1)

        # categorized normal
        categorized_preds, categorized_targets = categorize_data(targets, preds, peaks)
        acc, prec, rec, f1, cm = calculate_metrices(
            preds=categorized_preds,
            targets=categorized_targets,
            num_classes=num_classes,
            end_idxs=end_idxs,
        )
        acc_tab.append(acc)
        prec_tab.append(prec)
        rec_tab.append(rec)
        f1_tab.append(f1)
        cm_tab += cm

    print("Training!")
    print(
        f"Loss value: {running_loss/len(data_loader)}, Accuracy: {np.nanmean(acc_tab)}"
    )
    return_list = [
        running_loss / len(data_loader),
        np.nanmean(acc_tab),
        np.nanmean(prec_tab),
        np.nanmean(rec_tab),
        np.nanmean(f1_tab),
        cm_tab,
    ]

    return return_list
