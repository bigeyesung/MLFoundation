import numpy
import random


class RandomPLA():
    def __init__(self, dimension, count):
        self.__dimension = dimension
        self.__count = count

    def randomSet(self, path):
        training_set = open(path)
        random_list = []

        x_count = 0
        for line in training_set:
            x = []
            x.append(1)
            for str in line.split(' '):
                if len(str.split('\t')) == 1:
                    x.append(float(str))
                else:
                    x.append(float(str.split('\t')[0]))
                    x.append(int(str.split('\t')[1].strip()))
            random_list.append(x)
            x_count += 1
        random.shuffle(random_list)
        return random_list

    def trainSet(self, path):
        x_train = numpy.zeros((self.__count, self.__dimension))
        y_train = numpy.zeros((self.__count, 1))
        random_list = self.randomSet(path)
        for i in range(self.__count):
            for j in range(self.__dimension):
                x_train[i, j] = random_list[i][j]
            y_train[i] = random_list[i][self.__dimension]
        return x_train, y_train

    def countIteration(self, path):
        count = 0
        x_train, y_train = self.trainSet(path)
        w = numpy.zeros((self.__dimension, 1))
        while True:
            err = False
            for i in range(self.__count):
                if numpy.dot(x_train[i], w)[0] * y_train[i] <= 0:
                    w += y_train[i] * x_train[i].reshape(5, 1)
                    count += 1
                    err = 1
            if err == 0:
                break
        return count


if __name__ == '__main__':
    sum = 0
    for i in range(2000):
        perceptron = RandomPLA(5, 400)
        sum += perceptron.countIteration('source/hw1_15_train.dat')
    print(sum / 2000.0)