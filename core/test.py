"""Test script to classify target data."""

import torch
import torch.nn as nn

from utils import make_variable


def eval_tgt(encoder, classifier, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    encoder.eval()
    classifier.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():  
        for images, labels in data_loader:
            images = make_variable(images, requires_grad=False)
            labels = make_variable(labels, requires_grad=False).long()

            preds = classifier(encoder(images))
            loss = criterion(preds, labels)

            total_loss += loss.item()   
            pred_cls = preds.argmax(dim=1)  
            total_correct += pred_cls.eq(labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    avg_acc = total_correct / total_samples

    print("Avg Loss = {:.6f}, Avg Accuracy = {:.2%}".format(avg_loss, avg_acc))
