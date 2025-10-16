"""Pre-train encoder and classifier for source dataset."""

import torch
import torch.nn as nn
import torch.optim as optim

import params
from utils import make_variable, save_model


def train_src(encoder, classifier, data_loader):
    """Train classifier for source domain."""

    ####################
    # 1. setup network #
    ####################
    encoder.train()
    classifier.train()

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=params.c_learning_rate,
        betas=(params.beta1, params.beta2)
    )
    criterion = nn.CrossEntropyLoss()

    ####################
    # 2. train network #
    ####################
    for epoch in range(params.num_epochs_pre):
        for step, (images, labels) in enumerate(data_loader):
            images = make_variable(images)
            labels = make_variable(labels)

            optimizer.zero_grad()

            preds = classifier(encoder(images))
            loss = criterion(preds, labels)

            loss.backward()
            optimizer.step()

            if ((step + 1) % params.log_step_pre == 0):
                print("Epoch [{}/{}] Step [{}/{}]: loss={:.6f}".format(
                    epoch + 1,
                    params.num_epochs_pre,
                    step + 1,
                    len(data_loader),
                    loss.item()  
                ))

        # eval model on test set
        if ((epoch + 1) % params.eval_step_pre == 0):
            eval_src(encoder, classifier, data_loader)

        # save model parameters
        if ((epoch + 1) % params.save_step_pre == 0):
            save_model(encoder, f"ADDA-source-encoder-{epoch+1}.pt")
            save_model(classifier, f"ADDA-source-classifier-{epoch+1}.pt")

    # save final model
    save_model(encoder, "ADDA-source-encoder-final.pt")
    save_model(classifier, "ADDA-source-classifier-final.pt")

    return encoder, classifier


def eval_src(encoder, classifier, data_loader):
    """Evaluate classifier for source domain."""
    encoder.eval()
    classifier.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():  
        for images, labels in data_loader:
            images = make_variable(images)
            labels = make_variable(labels)

            preds = classifier(encoder(images))
            loss = criterion(preds, labels)
            total_loss += loss.item()

            pred_cls = preds.argmax(dim=1)  
            total_correct += pred_cls.eq(labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    avg_acc = total_correct / total_samples

    print("Avg Loss = {:.6f}, Avg Accuracy = {:.2%}".format(avg_loss, avg_acc))
