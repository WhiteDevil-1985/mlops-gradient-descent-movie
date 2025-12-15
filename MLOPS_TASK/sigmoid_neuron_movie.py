import numpy as np
from optimizers import (
    sigmoid,
    mse_loss,
    train_vanilla_gd,
    train_momentum_gd,
    train_nesterov_gd,
)

train_data = np.loadtxt("movie_train_data.csv", delimiter=",", skiprows=1)
X_train = train_data[:, :2]
y_train = train_data[:, 2]


test_data = np.loadtxt("movie_test_data.csv", delimiter=",", skiprows=1)
X_test = test_data[:, :2]
y_test = test_data[:, 2]

def predict_with_weights(X, w, b):
    z = X @ w + b
    y_hat = sigmoid(z)
    y_pred = (y_hat >= 0.5).astype(int)
    return y_hat, y_pred

def run_all_trainings():
    lr = 0.01
    epochs = 2000

    print("Training with vanilla GD...")
    w_gd, b_gd, losses_gd = train_vanilla_gd(X_train, y_train, lr=lr, epochs=epochs)
    print("Final vanilla GD loss:", losses_gd[-1])

    print("Training with Momentum GD...")
    w_mom, b_mom, losses_mom = train_momentum_gd(X_train, y_train, lr=lr, epochs=epochs, momentum=0.9)
    print("Final Momentum loss:", losses_mom[-1])

    print("Training with Nesterov GD...")
    w_nes, b_nes, losses_nes = train_nesterov_gd(X_train, y_train, lr=lr, epochs=epochs, momentum=0.9)
    print("Final Nesterov loss:", losses_nes[-1])

    np.savez("weights_gd.npz", w=w_gd, b=b_gd)
    np.savez("weights_mom.npz", w=w_mom, b=b_mom)
    np.savez("weights_nes.npz", w=w_nes, b=b_nes)

    return (w_gd, b_gd, losses_gd,
            w_mom, b_mom, losses_mom,
            w_nes, b_nes, losses_nes)

if __name__ == "__main__":
    (w_gd, b_gd, losses_gd,
     w_mom, b_mom, losses_mom,
     w_nes, b_nes, losses_nes) = run_all_trainings()

    y_hat_gd, y_pred_gd = predict_with_weights(X_test, w_gd, b_gd)
    test_loss_gd = mse_loss(y_hat_gd, y_test)
    print("Vanilla GD test loss:", test_loss_gd)

    y_hat_mom, y_pred_mom = predict_with_weights(X_test, w_mom, b_mom)
    test_loss_mom = mse_loss(y_hat_mom, y_test)
    print("Momentum GD test loss:", test_loss_mom)

    y_hat_nes, y_pred_nes = predict_with_weights(X_test, w_nes, b_nes)
    test_loss_nes = mse_loss(y_hat_nes, y_test)
    print("Nesterov GD test loss:", test_loss_nes)

    print("First 10 vanilla GD test predictions (class labels):")
    print(y_pred_gd[:10])
