import random
import math

class LogisticRegression:
    def __init__(self, X1, X2, Y, learningRate, threshold, w1=random.randint(0, 100), w2=random.randint(0, 100), w0=random.randint(0, 100)) -> None:
        self.w1 = w1
        self.w2 = w2
        self.w0 = w0
        self.X1 = X1
        self.X2 = X2
        self.Y = Y
        self.learningRate = learningRate
        self.threshold = threshold

    def predict(self, x1, x2):
        return 1/(1 + (math.e**-(x1*self.w1 + x2*self.w2 + self.w0)))
        
    def derivative_w1(self):
        total_error = 0
        for x1,x2,y in zip(self.X1, self.X2, self.Y):
            total_error += -((y*(1-self.predict(x1, x2)) + self.predict(x1, x2)*(1-y)) * x1)
        return total_error/len(self.X1)

    def derivative_w2(self):
        total_error = 0
        for x1,x2,y in zip(self.X1, self.X2, self.Y):
            total_error += -((y*(1-self.predict(x1, x2)) + self.predict(x1, x2)*(1-y)) * x2)
        return total_error/len(self.X2)

    def derivative_w0(self):
        total_error = 0
        for x1,x2,y in zip(self.X1, self.X2, self.Y):
            total_error += -((y*(1-self.predict(x1, x2)) + self.predict(x1, x2)*(1-y)))
        return total_error/len(X)

    def train(self):
        old = self.w1
        self.w1 = self.w1 - self.learningRate * self.derivative_w1()
        self.w2 = self.w2 - self.learningRate * self.derivative_w2()
        self.w0 = self.w0 - self.learningRate * self.derivative_w0()
        if abs(old - self.w1) < self.threshold: return False
        return True


if __name__ == "__main__":
    X1 = [3,2.5,1,2.5,2,1.5,0.5,1.75,0.25,1,0.25,0.20,0.15,2,1,0.15,0.10,0.5,1]
    X2 = [4,4,4,5,5,5,5,6,6,7,7,7,7,8,8,8,8,9,10]
    Y = [1,1,0,1,1,0,0,1,0,1,0,0,0,1,0,0,0,1,1]

    logisticRegression = LogisticRegression(X1, X2, Y, 0.0001, 0.001)

    cont = True
    while cont:
        cont = logisticRegression.train()
