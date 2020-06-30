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

if __name__ == "__main__":
    sol = Solution()
    # prepare train and test data
    train_x, train_y = sol.load_data("source/w4_train.dat")
    test_x, test_y = sol.load_data("source/w4_test.dat")

    # Q16-Q17
    Etrain_min = float("inf")
    Eval_min = float("inf")
    # value updated with Etrain_min
    Eout_Etrain_min = 0
    Eval_Etrain_min = 0
    optimal_lambda_Etrain_min = 0
    # value updated with Eval_min
    Etrain_Eval_min = 0
    Eout_Eval_min = 0
    optimal_lambda_Eval_min = 0
    split = 120
    for lambda_value in range(2, -11, -1):
        w_reg = sol.calculate_w_reg(train_x[:split], train_y[:split], pow(10, lambda_value))
        Etrain = sol.calculate_E(w_reg, train_x[:split], train_y[:split])
        Eval = sol.calculate_E(w_reg, train_x[split:], train_y[split:])
        Eout = sol.calculate_E(w_reg, test_x, test_y)
        if Etrain_min > Etrain:
            optimal_lambda_Etrain_min = lambda_value
            Etrain_min = Etrain
            Eout_Etrain_min = Eout
            Eval_Etrain_min = Eval
        if Eval_min > Eval:
            optimal_lambda_Eval_min = lambda_value
            Eval_min = Eval
            Eout_Eval_min = Eout
            Etrain_Eval_min = Etrain
    # Q16
    print('Q16: log10 = ', optimal_lambda_Etrain_min, ', Etrain= ', Etrain_min, ', Eval = ', Eval_Etrain_min,
        ', Eout = ', Eout_Etrain_min)
    # Q17
    print('Q17: log10 = ', optimal_lambda_Eval_min, ', Etrain= ', Etrain_Eval_min, ', Eval = ', Eval_min, ', Eout = ',
        Eout_Eval_min)
