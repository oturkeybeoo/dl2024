import math
a = 0
b = 0
L = 0.0001
threshold = 0.00001

def predict(x):
    return x*a + b

def square_error(x, y):
    return (y - predict(x)) ** 2

def mean_square_error(X, Y):
    total_error = 0
    for x,y in zip(X, Y):
        total_error += square_error(x, y)
    return total_error/len(X)

def root_mean_square_error(X, Y):
    return math.sqrt(mean_square_error(X, Y))

def square_error_derivative_a(x, y):
    return -2*(y - predict(x))*x

def mean_square_error_derivative_a(X, Y):
    total_error = 0
    for x,y in zip(X, Y):
        total_error += square_error_derivative_a(x, y)
    return total_error/len(X)

def square_error_derivative_b(x, y):
    return -2*(y - predict(x))

def mean_square_error_derivative_b(X, Y):
    total_error = 0
    for x,y in zip(X, Y):
        total_error += square_error_derivative_b(x, y)
    return total_error/len(X)

X = [30, 32.4138, 34.8276, 37.2414, 39.6552]
Y = [448.524, 509.248, 535.104, 551.432, 623.418]


if __name__ == "__main__":
    old_a = 100
    old_b = 100
    i = 1

    while abs(b - old_b) > threshold:
        old_a = a
        old_b = b
        a = a - L * mean_square_error_derivative_a(X, Y)
        b = b - L * mean_square_error_derivative_b(X, Y)
        i += 1
        print(i, root_mean_square_error(X, Y))
