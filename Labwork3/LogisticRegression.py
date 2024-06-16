import random
import math

class LogisticRegression:
    def __init__(self, X1, X2, Y, learningRate, threshold, w1=0, w2=0, w0=0) -> None:
        self.w1 = w1
        self.w2 = w2
        self.w0 = w0
        self.X1 = X1
        self.X2 = X2
        self.Y = Y
        self.learningRate = learningRate
        self.threshold = threshold

    def sigmoid(self, x1, x2):
        return 1/(1 + math.e**(-(x1*self.w1 + x2*self.w2 + self.w0)))
        
    def accuracy(self):
        count = 0
        for x1,x2,y in zip(self.X1, self.X2, self.Y):
            sigmoid = self.sigmoid(x1, x2)
            if sigmoid < 0.5: sigmoid = 0
            else: sigmoid = 1
            
            if sigmoid == y: count += 1
        return count/len(self.Y)

    def derivative_w1(self):
        total_error = 0
        for x1,x2,y in zip(self.X1, self.X2, self.Y):
            total_error += -((y*(1-self.sigmoid(x1, x2)) + self.sigmoid(x1, x2)*(1-y)) * x1)
        res = total_error/len(self.X1)
        return res

    def derivative_w2(self):
        total_error = 0
        for x1,x2,y in zip(self.X1, self.X2, self.Y):
            total_error += -((y*(1-self.sigmoid(x1, x2)) + self.sigmoid(x1, x2)*(1-y)) * x2)
        res = total_error/len(self.X1)
        return res

    def derivative_w0(self):
        total_error = 0
        for x1,x2,y in zip(self.X1, self.X2, self.Y):
            total_error += -((y*(1-self.sigmoid(x1, x2)) + self.sigmoid(x1, x2)*(1-y)))
        res = total_error/len(self.X1)
        return res

    def train(self):
        old = self.w1
        self.w1 = self.w1 - self.learningRate * self.derivative_w1()
        self.w2 = self.w2 - self.learningRate * self.derivative_w2()
        self.w0 = self.w0 - self.learningRate * self.derivative_w0()
        print(f"New w: {self.w1} {self.w2} {self.w0}")
        if abs(old - self.w1) < self.threshold: return False
        return True


if __name__ == "__main__":
    X1 = [3,2.5,1,2.5,2,1.5,0.5,1.75,0.25,1,0.25,0.20,0.15,2,1,0.15,0.10,0.5,1]
    X2 = [4,4,4,5,5,5,5,6,6,7,7,7,7,8,8,8,8,9,10]
    Y = [1,1,0,1,1,0,0,1,0,1,0,0,0,1,0,0,0,1,1]
    logisticRegression = LogisticRegression(X1, X2, Y, 1, 0.001)

    cont = True
    for i in range(10):
        logisticRegression.train()
        print(logisticRegression.accuracy())
