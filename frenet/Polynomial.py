import numpy as np
from scipy.linalg import solve

class Polynomial:
    def __init__(self, x0, v0, a0, xT=None, vT=None, aT=None, T=1.0):
        self.x0 = x0
        self.v0 = v0
        self.a0 = a0
        self.T = T

        if xT is not None:
            # Quintic Polynomial
            self.type = "quintic"
            self.xT = xT
            self.vT = vT
            self.aT = aT
            self.a1 = x0
            self.a2 = v0
            self.a3 = a0 / 2.0

            A = np.array([[T**3, T**4, T**5],
                          [3*T**2, 4*T**3, 5*T**4],
                          [6*T, 12*T**2, 20*T**3]])

            B = np.array([xT - self.a1 - self.a2 * T - self.a3 * T**2,
                          vT - self.a2 - 2 * self.a3 * T,
                          aT - 2 * self.a3])

            # Solve for a4, a5, a6
            self.a4, self.a5, self.a6 = solve(A, B)

        else:
            # Quartic Polynomial
            self.type = "quartic"
            self.vT = vT
            self.aT = aT
            self.a1 = x0
            self.a2 = v0
            self.a3 = a0 / 2.0

            A = np.array([[3*T**2, 4*T**3],
                          [6*T, 12*T**2]])

            B = np.array([vT - self.a2 - 2 * self.a3 * T,
                          aT - 2 * self.a3])

            # Solve for a4, a5
            self.a4, self.a5 = solve(A, B)
            self.a6 = 0  # No a6 for quartic polynomial

    def position(self, t):
        if self.type == "quintic":
            return self.a1 + self.a2 * t + self.a3 * t**2 + self.a4 * t**3 + self.a5 * t**4 + self.a6 * t**5
        return self.a1 + self.a2 * t + self.a3 * t**2 + self.a4 * t**3 + self.a5 * t**4

    def velocity(self, t):
        if self.type == "quintic":
            return self.a2 + 2 * self.a3 * t + 3 * self.a4 * t**2 + 4 * self.a5 * t**3 + 5 * self.a6 * t**4
        return self.a2 + 2 * self.a3 * t + 3 * self.a4 * t**2 + 4 * self.a5 * t**3

    def acceleration(self, t):
        if self.type == "quintic":
            return 2 * self.a3 + 6 * self.a4 * t + 12 * self.a5 * t**2 + 20 * self.a6 * t**3
        return 2 * self.a3 + 6 * self.a4 * t + 12 * self.a5 * t**2

    def jerk(self, t):
        if self.type == "quintic":
            return 6 * self.a4 + 24 * self.a5 * t + 60 * self.a6 * t**2
        return 6 * self.a4 + 24 * self.a5 * t


# class Polynomial:
#     def __init__(self, x0, v0, a0, xT=None, vT=None, aT=None, T=None):
#         if xT is not None and vT is not None and aT is not None:
#             # Quintic polynomial
#             self.type_ = "quintic"
#             self.x0_ = x0
#             self.v0_ = v0
#             self.a0_ = a0
#             self.xT_ = xT
#             self.vT_ = vT
#             self.aT_ = aT
#             self.T_ = T
#             self.a1_ = x0
#             self.a2_ = v0
#             self.a3_ = a0 / 2

#             A = np.array([
#                 [T**3, T**4, T**5],
#                 [3 * T**2, 4 * T**3, 5 * T**4],
#                 [6 * T, 12 * T**2, 20 * T**3]            
#             ])
#             B = np.array([
#                 xT - self.a1_ - self.a2_ * T - self.a3_ * T**2,
#                 vT - self.a2_ - 2 * self.a3_ * T,
#                 aT - 2 * self.a3_
#             ])
#             coefficients = np.linalg.solve(A, B)
#             self.a4_ = coefficients[0]
#             self.a5_ = coefficients[1]
#             self.a6_ = coefficients[2]
#         elif vT is not None and aT is not None:
#             # Quartic polynomial
#             self.type_ = "quartic"
#             self.x0_ = x0
#             self.v0_ = v0
#             self.a0_ = a0
#             self.vT_ = vT
#             self.aT_ = aT
#             self.T_ = T
#             self.a1_ = x0
#             self.a2_ = v0
#             self.a3_ = a0 / 2

#             A = np.array([
#                 [3 * T**2, 4 * T**3],
#                 [6 * T, 12 * T**2]
#             ])
#             B = np.array([
#                 vT - self.a2_ - 2 * self.a3_ * T,
#                 aT - 2 * self.a3_
#             ])
#             coefficients = np.linalg.solve(A, B)
#             self.a4_ = coefficients[0]
#             self.a5_ = coefficients[1]
#         else:
#             raise ValueError("Insufficient parameters for polynomial initialization")

#     def position(self, t):
#         if self.type_ == "quintic":
#             return (self.a1_ + self.a2_ * t + self.a3_ * t**2 +
#                     self.a4_ * t**3 + self.a5_ * t**4 + self.a6_ * t**5)
#         else:  # Quartic
#             return (self.a1_ + self.a2_ * t + self.a3_ * t**2 +
#                     self.a4_ * t**3 + self.a5_ * t**4)

#     def velocity(self, t):
#         if self.type_ == "quintic":
#             return (self.a2_ + 2 * self.a3_ * t + 3 * self.a4_ * t**2 +
#                     4 * self.a5_ * t**3 + 5 * self.a6_ * t**4)
#         else:  # Quartic
#             return (self.a2_ + 2 * self.a3_ * t + 3 * self.a4_ * t**2 +
#                     4 * self.a5_ * t**3)

#     def acceleration(self, t):
#         if self.type_ == "quintic":
#             return (2 * self.a3_ + 6 * self.a4_ * t +
#                     12 * self.a5_ * t**2 + 20 * self.a6_ * t**3)
#         else:  # Quartic
#             return (2 * self.a3_ + 6 * self.a4_ * t +
#                     12 * self.a5_ * t**2)

#     def jerk(self, t):
#         if self.type_ == "quintic":
#             return (6 * self.a4_ + 24 * self.a5_ * t + 60 * self.a6_ * t**2)
#         else:  # Quartic
#             return (6 * self.a4_ + 24 * self.a5_ * t)