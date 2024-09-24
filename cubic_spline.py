import numpy as np
import math
import bisect

class CubicSpline2D:
    def __init__(self, x, y):
        self.s = self.__calc_s(x, y)
        self.sx = self.__calc_spline(self.s, x)
        self.sy = self.__calc_spline(self.s, y)

    def __calc_spline(self, s, z):
        h = np.diff(s)
        if np.any(h < 0):
            raise ValueError("s coordinates must be sorted in ascending order")

        a, b, c, d = [], [], [], []

        # calc coefficient a
        a = [iz for iz in z]

        # calc coefficient c
        A = self.__calc_A(h)
        B = self.__calc_B(h, a)
        c = np.linalg.solve(A, B)

        # calc spline coefficient b and d
        for i in range(len(s) - 1):
            d_ = (c[i + 1] - c[i]) / (3.0 * h[i])
            b_ = 1.0 / h[i] * (a[i + 1] - a[i]) - h[i] / 3.0 * (2.0 * c[i] + c[i + 1])
            d.append(d_)
            b.append(b_)

        return a, b, c, d, s

    def __calc_s(self, x, y):   # chuoi khoang cach tich luy giua cac diem
        dx = np.diff(x)
        dy = np.diff(y)
        ds = np.hypot(dx, dy)
        s = [0]
        s.extend(np.cumsum(ds))
        return s

    def calc_position(self, s):
        x = self.__calc_spline_value(self.sx, s)
        y = self.__calc_spline_value(self.sy, s)

        return x, y

    def __calc_spline_value(self, spline, s):
        a, b, c, d, s_vals = spline
        if s < s_vals[0] or s > s_vals[-1]:
            return None
        i = bisect.bisect(s_vals, s) - 1
        ds = s - s_vals[i]
        return a[i] + b[i] * ds + c[i] * ds ** 2.0 + d[i] * ds ** 3.0

    def calc_curvature(self, s):
        dx = self.__calc_spline_derivative(self.sx, s, 1)
        ddx = self.__calc_spline_derivative(self.sx, s, 2)
        dy = self.__calc_spline_derivative(self.sy, s, 1)
        ddy = self.__calc_spline_derivative(self.sy, s, 2)
        k = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2) ** (3 / 2))
        return k

    def calc_yaw(self, s):
        dx = self.__calc_spline_derivative(self.sx, s, 1)
        dy = self.__calc_spline_derivative(self.sy, s, 1)
        yaw = math.atan2(dy, dx)
        return yaw

    def __calc_spline_derivative(self, spline, s, order):
        a, b, c, d, s_vals = spline
        if s < s_vals[0] or s > s_vals[-1]:
            return None
        i = bisect.bisect(s_vals, s) - 1
        ds = s - s_vals[i]
        if order == 1:
            return b[i] + 2.0 * c[i] * ds + 3.0 * d[i] * ds ** 2.0
        elif order == 2:
            return 2.0 * c[i] + 6.0 * d[i] * ds
        return None

    def __calc_A(self, h):
        nx = len(h) + 1
        A = np.zeros((nx, nx))
        A[0, 0] = 1.0
        for i in range(nx - 1):
            if i != (nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[-1, -2] = 0.0
        A[-1, -1] = 1.0
        return A

    def __calc_B(self, h, a):
        nx = len(h) + 1
        B = np.zeros(nx)
        for i in range(nx - 2):
            B[i + 1] = 3.0 * (a[i + 2] - a[i + 1]) / h[i + 1] - 3.0 * (a[i + 1] - a[i]) / h[i]
        return B