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

    def feature_transform(self,features):
        new = np.zeros((len(features), 6))
        new[:, 0:3] = features[:, :] * 1
        new[:, 3] = features[:, 1] * features[:, 2]
        new[:, 4] = features[:, 1] * features[:, 1]
        new[:, 5] = features[:, 2] * features[:, 2]
        return new


if __name__ == '__main__':
    # 14 in_sample_err
    sol = Solution()
    (features, labels) = sol.training_data_with_random_error(1000)
    new_feature = sol.feature_transform(features)
    wlin_default = sol.linear_regression_closed_form(new_feature, labels)
    min_error_in = float("inf")

    for i in range(1000):
        (features, labels) = sol.training_data_with_random_error(1000)
        new_feature = sol.feature_transform(features)
        wlin = sol.linear_regression_closed_form(new_feature, labels)
        in_sample_err = sol.in_sample_err(new_feature, labels, wlin)
        if in_sample_err <= min_error_in:
            wlin_default = wlin
            min_error_in = in_sample_err
    print("W linear: ", wlin_default) 

    error_out = []
    for i in range(1000):
        (features, labels) = sol.training_data_with_random_error(1000)
        new_features = sol.feature_transform(features)
        error_out.append(sol.in_sample_err(new_features, labels, wlin_default))
    print("Average of E_out is: ", sum(error_out) / (len(error_out) * 1.0))
 
