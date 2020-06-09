import random
import numpy as np


# target function f(x1, x2) = sign(x1^2 + x2^2 - 0.6)
def target_function(x1, x2):
    if (x1 * x1 + x2 * x2 - 0.6) >= 0:
        return 1
    else:
        return -1


# create train_set
def training_data_with_random_error(num=1000):
    features = np.zeros((num, 3))
    labels = np.zeros((num, 1))

    points_x1 = np.array([round(random.uniform(-1, 1), 2) for i in range(num)])
    points_x2 = np.array([round(random.uniform(-1, 1), 2) for i in range(num)])

    for i in range(num):
        # create random feature
        features[i, 0] = 1
        features[i, 1] = points_x1[i]
        features[i, 2] = points_x2[i]
        labels[i] = target_function(points_x1[i], points_x2[i])
        # choose 10% error labels
        if i <= num * 0.1:
            if labels[i] < 0:
                labels[i] = 1
            else:
                labels[i] = -1
    return features, labels


def error_rate(features, labels, w):
    wrong = 0
    for i in range(len(labels)):
        if np.dot(features[i], w) * labels[i, 0] < 0:
            wrong += 1
    return wrong / (len(labels) * 1.0)


def linear_regression_closed_form(X, Y):
    """
        linear regression:
        model     : g(x) = Wt * X
        strategy  : squared error
        algorithm : close form(matrix)
        result    : w = (Xt.X)^-1.Xt.Y
    """
    return np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(Y)


if __name__ == '__main__':

    # 13
    error_rate_array = []
    for i in range(1000):
        (features, labels) = training_data_with_random_error(1000)
        w13 = linear_regression_closed_form(features, labels)
        error_rate_array.append(error_rate(features, labels, w13))

    # error rate, approximately 0.5
    avr_err = sum(error_rate_array) / (len(error_rate_array) * 1.0)

    print("13--Linear regression for classification without feature transform:Average error--", avr_err)
