import os
import time
import copy

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)


def is_data_augmented():
    return os.path.isdir("augmented_data")


def train_model(model,
                dataloaders,
                dataset_sizes,
                loss_func,
                optimizer,
                scheduler,
                device,
                num_epochs=30,
                start_epoch=1,
                history=None):

    start_time = time.time()

    if history is None:
        history = {
            'train': {
                'loss': [],
                'accuracy': [],
                'topK_accuracy': []
            },
            'valid': {
                'loss': [],
                'accuracy': [],
                'topK_accuracy': []
            }
        }

    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_topK_acc = 0.0

    for epoch in range(start_epoch, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # training mode
            else:
                model.eval()  # evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_topK_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)  # score
                    _, predicts = torch.max(outputs, 1)  # batch predictions
                    loss = loss_func(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(predicts == labels.data)
                running_topK_corrects += topK_accuracy(outputs, labels, topK=5)

            # epoch statistics
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_topK_acc = running_topK_corrects.double() / dataset_sizes[phase]

            if phase == 'valid':
                scheduler.step(epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f} TopK-Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_topK_acc))
            history[phase]['loss'].append(epoch_loss)
            history[phase]['accuracy'].append(epoch_acc)
            history[phase]['topK_accuracy'].append(epoch_topK_acc)

            # find best top-1 and top-k accuracy and deep copy the model based on the best top-1 accuracy
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

            if phase == 'valid' and epoch_topK_acc > best_topK_acc:
                best_topK_acc = epoch_topK_acc

        print()

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val TopK-Acc: {:4f}'.format(best_topK_acc))

    # return model with its best weights and historical data
    model.load_state_dict(best_model_weights)
    return model, history


def topK_accuracy(outputs, labels, topK=1):
    _, pred = outputs.topk(topK, dim=1)
    pred = pred.t()
    corrects = pred.eq(labels.view(1, -1).expand_as(pred))
    corrects = corrects.contiguous().view(-1).float().sum(0, keepdim=True)
    return corrects[0]


def plot_loss_and_accuracy(history):
    fig, ax = plt.subplots(3, 1)
    fig.set_size_inches(8, 8)
    plt.tight_layout()

    num_epochs = len(history['train']['loss'])

    ax[0].plot(np.arange(start=1, stop=num_epochs + 1), history['train']['loss'], color='b', label='Training')
    ax[0].plot(np.arange(start=1, stop=num_epochs + 1), history['valid']['loss'], color='r', label='Validation')
    ax[0].set_title('Loss')
    ax[0].legend()
    ax[0].xaxis.set_major_locator(MultipleLocator(2))
    ax[0].xaxis.set_major_formatter(FormatStrFormatter('%d'))

    ax[1].plot(np.arange(start=1, stop=num_epochs + 1), history['train']['accuracy'], color='b', label='Training')
    ax[1].plot(np.arange(start=1, stop=num_epochs + 1), history['valid']['accuracy'], color='r', label='Validation')
    ax[1].set_title('Accuracy')
    ax[1].legend()
    ax[1].xaxis.set_major_locator(MultipleLocator(2))
    ax[1].xaxis.set_major_formatter(FormatStrFormatter('%d'))

    ax[2].plot(np.arange(start=1, stop=num_epochs + 1), history['train']['topK_accuracy'], color='b', label='Training')
    ax[2].plot(np.arange(start=1, stop=num_epochs + 1), history['valid']['topK_accuracy'], color='r', label='Validation')
    ax[2].set_title('Top-5 Accuracy')
    ax[2].legend()
    ax[2].xaxis.set_major_locator(MultipleLocator(2))
    ax[2].xaxis.set_major_formatter(FormatStrFormatter('%d'))
