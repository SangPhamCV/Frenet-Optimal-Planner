import numpy as np

class Polynomial:
    def __init__(self, xs, vxs, axs, xe=None, vxe=None, axe=None, time=None):
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        if xe is not None and vxe is not None and axe is not None:
            A = np.array([[time ** 3, time ** 4, time ** 5],
                          [3 * time ** 2, 4 * time ** 3, 5 * time ** 4],
                          [6 * time, 12 * time ** 2, 20 * time ** 3]])
            b = np.array([xe - self.a0 - self.a1 * time - self.a2 * time ** 2,
                          vxe - self.a1 - 2 * self.a2 * time,
                          axe - 2 * self.a2])
            x = np.linalg.solve(A, b)
            self.a3 = x[0]
            self.a4 = x[1]
            self.a5 = x[2]
        elif vxe is not None and axe is not None:
            A = np.array([[3 * time ** 2, 4 * time ** 3],
                          [6 * time, 12 * time ** 2]])
            b = np.array([vxe - self.a1 - 2 * self.a2 * time,
                          axe - 2 * self.a2])
            x = np.linalg.solve(A, b)
            self.a3 = x[0]
            self.a4 = x[1]
            self.a5 = 0

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5
        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4
        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3
        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2
        return xt