import random
import numpy as np

class Solution():
    # target function f(x1, x2) = sign(x1^2 + x2^2 - 0.6)
    def target_function(self, x1, x2):
        if (x1 * x1 + x2 * x2 - 0.6) >= 0:
            return 1
        else:
            return -1


    # create train_set
    def training_data_with_random_error(self, num=1000):
        features = np.zeros((num, 3))
        labels = np.zeros((num, 1))

        points_x1 = np.array([round(random.uniform(-1, 1), 2) for i in range(num)])
        points_x2 = np.array([round(random.uniform(-1, 1), 2) for i in range(num)])

        for i in range(num):
            # create random feature
            features[i, 0] = 1
            features[i, 1] = points_x1[i]
            features[i, 2] = points_x2[i]
            labels[i] = self.target_function(points_x1[i], points_x2[i])
            # choose 10% error labels
            if i <= num * 0.1:
                if labels[i] < 0:
                    labels[i] = 1
                else:
                    labels[i] = -1
        return features, labels


    def in_sample_err(self, features, labels, w):
        wrong = 0
        for i in range(len(labels)):
            if np.dot(features[i], w) * labels[i, 0] < 0:
                wrong += 1
        return wrong / (len(labels) * 1.0)


    def linear_regression_closed_form(self, X, Y):
        """
            linear regression:
            model     : g(x) = Wt * X
            strategy  : squared error
            algorithm : close form(matrix)
            result    : Wlin = (Xt.X)^-1.Xt.Y (pseudo-inverse)
        """
        return np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(Y)


if __name__ == '__main__':
    # 13 in_sample_err
    in_sample_err_array = []
    sol = Solution()
    for i in range(1000):
        (features, labels) = sol.training_data_with_random_error(1000)
        w13 = sol.linear_regression_closed_form(features, labels)
        in_sample_err_array.append(sol.in_sample_err(features, labels, w13))

    # in_sample_err rate, approximately 0.5
    avr_err = sum(in_sample_err_array) / (len(in_sample_err_array) * 1.0)

    print("13--Linear regression for classification without feature transform:Average error--", avr_err)
