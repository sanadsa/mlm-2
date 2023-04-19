# Student name: Sanad Satel
# Student ID: 208946533
# Student name: Yaakov Kourdi
# Student ID: 311400238

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Define the regression quadratic model
def quadratic_model(x, w):
    return w[0] + w[1]*x + w[2]*x**2


# Define the empirical risk function
def empirical_risk(x, y, w):
    y_pred = quadratic_model(x, w)
    loss = lambda y, y_pred: (y - y_pred)**2 # the mean squared error between the model's predicNons and the true labels
    return np.mean(loss(y, y_pred))

# Define the gradient of the empirical risk function
def gradient(x, y, w):
    y_pred = quadratic_model(x, w)
    d_w0 = np.mean(2*(y_pred - y))
    d_w1 = np.mean(2*(y_pred - y)*x)
    d_w2 = np.mean(2*(y_pred - y)*x**2)
    return np.array([d_w0, d_w1, d_w2])

# Define the gradient descent algorithm
def gradient_descent(x, y, w_init, lr, num_iterations):
    w = w_init
    for i in range(num_iterations):
        grad = gradient(x, y, w)
        w = w - lr*grad
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
    num_iterations = 10

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


#Debugging:
# you may see that the empirical risk is increasing. What could be the problem?
    # Learning rate is too large: 
    #  If the learning rate is too large, the weight updates during each iteration will be too large,
    #  and the algorithm may overshoot the optimal solution, causing the empirical risk to increase instead of decrease.
    #  In this case, we may need to reduce the learning rate to allow for smaller weight updates and better convergence.

    # Learning rate is too small: 
    #  If the learning rate is too small, the weight updates during each iteration will be too small,
    #  and the algorithm may take a long time to converge, or may get stuck in a local minimum.
    #  In this case, we may need to increase the learning rate to allow for larger weight updates and faster convergence.


# when decreasing the learning rate, the empirical risk decreases and the weights are closer to the optimal solution.

# the hypothesis (fitted model) is not close enough to the training data. What actions could you take to improve the fitting?
    # 1. Increase the number of iterations
    # 2. Increase the learning rate
    # 3. Increase the training data by collecting more data
    # 4. Increase the complexity of the model (e.g. add more features)

