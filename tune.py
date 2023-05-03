import datetime

import numpy as np

from train import main as train_network


def get_param():
    """
    Returns a random number sampled logarithmically from 0.001 to 0.1
    """
    return np.power(10, -2 * np.random.rand() - 1)


def hparam_optimization():
    """
    Does a simple random search hyperparameter optimization
    """

    num_trials = 10
    for i in range(num_trials):
        # Sample the hyperparameters
        lr = get_param()
        alpha = get_param()
        beta = get_param()

        epochs = 14

        now = datetime.datetime.now()
        date_string = now.strftime("%Y%m%d%H%M%S")

        name = (
            date_string
            + f"_epochs_{epochs}_lr_{lr:.2f}_alpha_{alpha:.2f}_beta{beta:.2f}"
        )

        print("Training: " + name)
        train_network(name, lr_init=lr, alpha=alpha, beta=beta, num_epochs=epochs)


if __name__ == "__main__":
    hparam_optimization()
