import numpy as np
class Solution:

    # load data
    def load_data(self, filename):
        code = open(filename, "r")
        lines = code.readlines()
        xn = np.zeros((len(lines), 3)).astype(np.float)
        yn = np.zeros((len(lines),)).astype(np.int)

        for i in range(0, len(lines)):
            line = lines[i]
            line = line.rstrip('\r\n').replace('\t', ' ').split(' ')
            xn[i, 0] = 1
            for j in range(1, len(xn[0])):
                xn[i, j] = float(line[j - 1])
            yn[i] = int(line[len(xn[0]) - 1])
        return xn, yn


    # W_reg:
    def calculate_w_reg(self, x, y, lambda_value):
        # W_reg = Inv(x.T *x + L*I) * x.T*y
        return np.dot(np.dot(np.linalg.inv(np.dot(x.transpose(), x) + lambda_value * np.eye(x.shape[1])), x.transpose()), y)


    # test result
    def calculate_E(self, w, x, y):
        scores = np.dot(w, x.transpose())
        predicts = np.where(scores >= 0, 1.0, -1.0)
        E_out_num = sum(predicts != y)
        return (E_out_num * 1.0) / predicts.shape[0]


if __name__ == '__main__':
    # prepare train and test data
    sol = Solution()
    train_x, train_y = sol.load_data("source/w4_train.dat")
    test_x, test_y = sol.load_data("source/w4_test.dat")

    # Q13
    lambda_value = 10
    W = sol.calculate_w_reg(train_x, train_y, lambda_value)
    Ein = sol.calculate_E(W, train_x, train_y)
    Eout = sol.calculate_E(W, test_x, test_y)
    print('Q13: Ein = ', Ein, ', Eout= ', Eout)
