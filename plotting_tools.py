import matplotlib.pyplot as plt
import numpy as np
import torch


# TALK ABOUT HOW WITH MORE TIME I WOULD INCORPORATE THESE INTO NN MODEL CLASSES

def plot_randomsample(dataset, label_map: dict, nrows=2, ncols=5, title="Random Sample of Dataset") -> plt.figure:
    figure = plt.figure(figsize=(3*ncols,3*nrows))
    figure.suptitle(title)
    
    for i in range(1, ncols * nrows + 1):
        sample_ind = np.random.randint(0, len(dataset))
        x, y = dataset[sample_ind]
        figure.add_subplot(nrows, ncols, i)
        plt.title(label_map[y])
        plt.axis('off')
        plt.imshow(x.squeeze(), cmap='gray')

    return figure


def plot_predictions(model, test_data, label_map: dict, device="cpu") -> plt.figure:
    figure = plt.figure(figsize=(14,6))
    figure.suptitle("Predictions on Sample of Test Set \n (Truth, Pred)")
    cols, rows = 5, 2

    display_batch = torch.utils.data.DataLoader(test_data, batch_size=10, shuffle=True)
    display_x, display_y = next(iter(display_batch))

    y_preds = torch.max(model(display_x.to(device)), 1)[1]

    for i in range(0, cols * rows):
        
        x, y = display_x[i], display_y[i]

        y_pred = y_preds[i]
        figure.add_subplot(rows, cols, i + 1)
        plt.title("(%s, %s)" % (label_map[y.item()], label_map[y_pred.item()]))
        plt.axis('off')
        plt.imshow(x.squeeze(), cmap='gray')

    return figure


def plot_perturbance(model, test_data, uap, label_map: dict, device="cpu") -> plt.figure:
    figure = plt.figure(figsize=(14,6))
    figure.suptitle("UAP Applied to MLP: Clean (Top) vs. Perturbed (Bottom) \n (Truth, Pred)")
    cols, rows = 5, 2

    test_batch = torch.utils.data.DataLoader(test_data, batch_size=5, shuffle=True)
    X, Y_true = next(iter(test_batch))
    Xhat_np = np.array(X) + uap

    # X = X.to(device)
    Xhat = torch.tensor(Xhat_np).to(device)
    Y_preds = torch.max(model(X.to(device)), 1)[1].to(device)
    Yhat_preds = torch.max(model(Xhat), 1)[1].to(device)

    for i in range(0, cols):
        x, y = X[i], Y_true[i]
        xhat = Xhat_np[i, :, :, :]

        y_pred = Y_preds[i]
        figure.add_subplot(rows, cols, i + 1)
        plt.title("(%s, %s)" % (label_map[y.item()], label_map[y_pred.item()]))
        plt.axis('off')
        plt.imshow(x.squeeze(), cmap='gray')

        yhat_pred = Yhat_preds[i]
        figure.add_subplot(rows, cols, i + 6)
        plt.title("(%s, %s)" % (label_map[y.item()], label_map[yhat_pred.item()]))
        plt.axis('off')
        plt.imshow(xhat.squeeze(), cmap='gray')

    return figure