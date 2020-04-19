import numpy

class PLA():
    def __init__(self, dimension, count):
        self.__dimension = dimension
        self.__count = count

    # get data
    def trainSet(self, path):
        training_set = open(path)
        x_train = numpy.zeros((self.__count, self.__dimension))
        y_train = numpy.zeros((self.__count, 1))
        x_count = 0
        for line in training_set:
            tmpx = []
            # add 1 dimension manually
            tmpx.append(1)
            for data in line.split(' '):
                if len(data.split('\t')) == 1:
                    tmpx.append(float(data))
                else:
                    tmpx.append(float(data.split('\t')[0]))
                    y_train[x_count] = int(data.split('\t')[1].strip())
            x_train[x_count] = tmpx

            x_count += 1
        return x_train, y_train

    def countIteration(self, path):
        count = 0
        x_train, y_train = self.trainSet(path)
        w = numpy.zeros((self.__dimension, 1))
        # loop until all x are classified right
        while True:
            err = False
            for ind in range(self.__count):
                # if sign(w,x) != y:
                #   w = w + ▵w(y*x)
                sign_result = numpy.dot(x_train[ind], w)
                if sign_result[0]*y_train[ind]<=0:
                    w = w + y_train[ind]* x_train[ind].reshape(5,1)
                    count += 1
                    err = True
            # stop until no err
            if not err:
                break
        return count


if __name__ == '__main__':
    perceptron = PLA(5, 400)
    print(perceptron.countIteration("source/hw1_15_train.dat"))
