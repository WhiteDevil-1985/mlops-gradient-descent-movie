# MLOps Gradient Descent Movie Task

This project trains a single sigmoid neuron to predict whether a person likes a movie based on:
- `action_level`
- `romance_level`

Three optimization algorithms are implemented:
- Vanilla Gradient Descent
- Momentum Gradient Descent
- Nesterov Gradient Descent

## Files

- `optimizers.py` – defines the sigmoid function, mean squared error loss, gradient computation, and the three training functions.
- `sigmoid_neuron_movie.py` – loads `movie_train_data.csv` and `movie_test_data.csv`, trains the neuron with all three algorithms, reports losses, and runs inference on the test data. [file:2][file:3]
- `movie_train_data.csv` – training dataset with columns `action_level, romance_level, like_movie`. [file:2]
- `movie_test_data.csv` – test dataset with the same columns, used for inference. [file:3]

## How to run

1. Open a terminal in the project folder.
2. Make sure NumPy is installed:

    python3 -m pip install numpy

3. Run the training and inference script:
   
    python3 sigmoid_neuron_movie.py

