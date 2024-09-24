import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb
from typing import List, Tuple, Optional

def binomial_coefficient(n: int, i: int) -> int:
    """Calculate the binomial coefficient C(n, i)."""
    return comb(n, i, exact=True)

def bezier_curve(points: np.ndarray, t: float) -> np.ndarray:
    """Calculate the position of a Bézier curve at time t."""
    n = len(points) - 1
    b_t = sum(binomial_coefficient(n, i) * ((1 - t) ** (n - i)) * (t ** i) * point
              for i, point in enumerate(points))
    return b_t

def compute_bezier_points(control_points: np.ndarray, num_points: int = 100) -> np.ndarray:
    """Compute points on a Bézier curve."""
    t_values = np.linspace(0, 1, num_points)
    return np.array([bezier_curve(control_points, t) for t in t_values])

def plot_bezier_curve(control_points: np.ndarray, curve_points: np.ndarray, highlight_indices: Optional[List[int]] = None) -> None:
    """Plot a Bézier curve with control points and highlighted points."""
    plt.figure()
    
    # Plot control points and connecting lines
    plt.plot(control_points[:, 0], control_points[:, 1], 'ko--', label='Control Points')
    plt.plot(curve_points[:, 0], curve_points[:, 1], 'b-', label='Bezier Curve')

    if highlight_indices:
        highlight_coords = curve_points[highlight_indices]
        plt.scatter(highlight_coords[:, 0], highlight_coords[:, 1], color='red', zorder=5)
        for i, (x, y) in enumerate(highlight_coords):
            plt.text(x, y, f'P{highlight_indices[i]}', fontsize=9, ha='right')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Bezier Curve with Highlighted Points')
    plt.legend()
    plt.grid(True)
    plt.show()

def main(control_points: List[Tuple[float, float]], highlight_indices: Optional[List[int]] = None) -> None:
    """Main function to compute and plot a Bézier curve."""
    # Convert control points to numpy array
    control_points_np = np.array(control_points)
    
    # Compute Bezier curve points
    bezier_points = compute_bezier_points(control_points_np)
    
    # Plot Bezier curve
    plot_bezier_curve(control_points_np, bezier_points, highlight_indices)

if __name__ == "__main__":
    # Define control points
    control_points = [(0.2, 0), (0, 1), (0.6, 1), (1, 0), (1.4, 1)]
    highlight_indices = [0, 19, 39, 59, 79, -1]

    main(control_points, highlight_indices)

