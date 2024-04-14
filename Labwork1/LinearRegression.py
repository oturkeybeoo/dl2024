import random

class LinearRegression:
    def __init__(self, X, Y, learningRate, threshold, init_a=random.randint(0, 100), init_b=random.randint(0, 100)) -> None:
        self.a = init_a
        self.b = init_b
        self.X = X
        self.Y = Y
        self.learningRate = learningRate
        self.threshold = threshold

    def predict(self, x):
        return x*self.a + self.b

    def mean_square_error(self):
        total_error = 0
        for x,y in zip(self.X, self.Y):
            total_error += (y - self.predict(x)) ** 2
        return total_error/len(X)

    def derivative_a(self):
        total_error = 0
        for x,y in zip(self.X, self.Y):
            total_error += -2*(y - self.predict(x))*x
        return total_error/len(X)

    def derivative_b(self):
        total_error = 0
        for x,y in zip(self.X, self.Y):
            total_error += -2*(y - self.predict(x))
        return total_error/len(X)

    def train(self):
        old_a = self.a
        self.a = self.a - self.learningRate * self.derivative_a()
        self.b = self.b - self.learningRate * self.derivative_b()
        if abs(old_a - self.a) < self.threshold: return False
        return True


if __name__ == "__main__":
    X = [30, 32.4138, 34.8276, 37.2414, 39.6552]
    Y = [448.524, 509.248, 535.104, 551.432, 623.418]

    linearRegression = LinearRegression(X, Y, 0.0001, 0.001)

    cont = True
    while cont:
        cont = linearRegression.train()
        print(linearRegression.mean_square_error())