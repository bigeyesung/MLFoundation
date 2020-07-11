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


    # W_reg = Inv(x.T *x + L*I) * x.T*y
    def calculate_w_reg(self, x, y, lambda_value):
        return np.dot(np.dot(np.linalg.inv(np.dot(x.transpose(), x) + lambda_value * np.eye(x.shape[1])), x.transpose()), y)


    # test result
    def calculate_E(self, w, x, y):
        scores = np.dot(w, x.transpose())
        predicts = np.where(scores >= 0, 1.0, -1.0)
        E_out_num = sum(predicts != y)
        return (E_out_num * 1.0) / predicts.shape[0]


if __name__ == '__main__':
    sol = Solution()
    # prepare train and test data
    train_x, train_y = sol.load_data("source/w4_train.dat")
    test_x, test_y = sol.load_data("source/w4_test.dat")

    # Q14-Q15
    Ein_min = float("inf")
    optimal_Eout = 0
    optimal_lambda_Ein = 0
    Eout_min = float("inf")
    optimal_Ein = 0
    optimal_lambda_Eout = 0
    # we choose lamda from large numbers
    for lambda_value in range(2, -11, -1):
        # calculate ridge regression W
        w_reg = sol.calculate_w_reg(train_x, train_y, pow(10, lambda_value))
        Ein = sol.calculate_E(w_reg, train_x, train_y)
        Eout = sol.calculate_E(w_reg, test_x, test_y)
        # update Ein,Eout,lambda
        if Ein_min > Ein:
            Ein_min = Ein
            optimal_lambda_Ein = lambda_value
            optimal_Eout = Eout
        if Eout_min > Eout:
            Eout_min = Eout
            optimal_lambda_Eout = lambda_value
            optimal_Ein = Ein
    # Q14
    print('Q14: log10lambda = ', optimal_lambda_Ein, ', Ein= ', Ein_min, ', Eout = ', optimal_Eout)
    # Q15
    print('Q15: log10lambda = ', optimal_lambda_Eout, ', Ein = ', optimal_Ein, ', Eout= ', Eout_min)

    