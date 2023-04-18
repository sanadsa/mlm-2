# Student name: ...
# Student ID: ...


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Define the regression quadratic model
def quadratic_model(x, w):
    return w[0] + w[1]*x + ...


# Define the empirical risk function
def empirical_risk(x, y, w):
    y_pred = ...
    return np.mean(...)

# Define the gradient of the empirical risk function
def gradient(x, y, w):
    y_pred = quadratic_model(x, w)
    d_w0 = np.mean(2*(y_pred - y))
    d_w1 = ...
    d_w2 = ...
    return np.array([d_w0, d_w1, d_w2])


# Define the gradient descent algorithm
def gradient_descent(x, y, w_init, lr, num_iterations):
    w = w_init
    for i in range(...):
        grad = gradient(x, y, w)
        w = w - ...
        risk = empirical_risk(x, y, w)
        print(f"Iteration {i}: Empirical risk={risk:.4f}, w={w}")
    return w


if __name__ == '__main__':
    
    data = pd.read_csv('data.csv')
    x = data['x'].values
    y = data['y'].values

    # Split the dataset into a training set and a validation set
    num_train = int(0.8*len(x))
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = x[num_train:], y[num_train:]

    plt.scatter(x_train, y_train, label='Training data')
    plt.show()

    # Train the model on the training set
    w_init = np.array([0, 0, 0])
    lr = 0.01
    num_iterations = 100

    w_star = gradient_descent(x_train, y_train, w_init, lr, num_iterations)

    # Evaluate the model on the validation set
    risk_val = empirical_risk(x_val, y_val, w_star)
    print(f"Validation risk: {risk_val:.4f}")

    # Plot the training data and the estimated hypothesis
    plt.scatter(x_train, y_train, label='Training data')
    x_plot = np.linspace(np.min(x_train), np.max(x_train), 100)
    y_plot = quadratic_model(x_plot, w_star)
    plt.plot(x_plot, y_plot, label='Estimated hypothesis')
    plt.legend()
    plt.show()

