class GradientDescent:
    def __init__(self, L, threshold, x0, a=0, b=0, c=0) -> None:
        self.step = 0
        self.L = L
        self.threshold = threshold
        self.old_x = x0
        self.new_x = x0
        self.a = a
        self.b = b
        self.c = c
        self.d_a = 0
        self.d_b = 0

    def derivative(self):
        self.d_a = 2 * self.a
        self.d_b = self.b
        self.print_derivative()
        return True

    def print_derivative(self):
        res = "The derivative of "
        if self.a != 0: res += f"{self.a}x^2 "
        if self.b != 0: res += f"+ {self.b}x "
        if self.c != 0: res += f"+ {self.c} "
        res += "is "
        if self.d_a != 0: res += f"{self.d_a}x "
        if self.d_b != 0: res += F"+ {self.d_b}" 
        print(res)

    # Input the function a*x^2 + b*x + c and the initial value x0
    def get(self):
        self.old_x = self.new_x
        self.new_x = self.old_x - self.L * (self.d_a * self.old_x + self.d_b)
        self.step += 1
        if abs(self.new_x - self.old_x) < self.threshold: return False
        return True
    
    def show(self):
        print(self.step, self.new_x, self.a*self.new_x**2+self.b*self.new_x+self.c)


if __name__ == "__main__":
    gd = GradientDescent(L=0.4, threshold=0.01, x0=-10, a=1)
    gd.derivative()

    cont = True
    while cont:
        cont = gd.get()
        gd.show()