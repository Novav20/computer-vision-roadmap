# rbf_interpolator.py

import numpy as np
import scipy.linalg # For solving linear system, more robust than np.linalg.solve for some cases
# import numpy.linalg # Alternative
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # For 3D plotting
import os

# --- Configuration ---
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_week5_tuesday_rbf")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- RBF Basis Functions ---
def gaussian_rbf(r, epsilon):
    # Epsilon is the shape parameter (width of Gaussian)
    return np.exp(-(epsilon * r)**2)

def multiquadric_rbf(r, epsilon):
    # Epsilon is a shape parameter
    return np.sqrt((epsilon * r)**2 + 1) # Common form, can also be sqrt(r^2 + epsilon^2)

def inverse_multiquadric_rbf(r, epsilon):
    return 1.0 / multiquadric_rbf(r, epsilon)

def linear_rbf(r, epsilon=None): # Epsilon not used but kept for consistent signature
    return r

def thin_plate_spline_rbf(r, epsilon=None): # Epsilon not used
    # Avoid log(0) by checking r
    # Note: True TPS for 2D includes a linear polynomial part by default for conditional positive definiteness
    # This basic RBF part is often used without the explicit polynomial part in simpler implementations
    # if the system matrix is well-conditioned or regularized.
    # If r is very small (e.g., 0), r^2 log(r) -> 0.
    # We add a small constant to avoid log(0) if r can be exactly 0.
    # For r=0, phi(0) = 0.
    r_safe = np.where(r == 0, 1e-10, r) # Avoid log(0)
    return (r_safe**2) * np.log(r_safe)


RBF_FUNCTIONS = {
    'gaussian': gaussian_rbf,
    'multiquadric': multiquadric_rbf,
    'inverse_multiquadric': inverse_multiquadric_rbf,
    'linear': linear_rbf,
    'thin_plate_spline': thin_plate_spline_rbf
}

# --- RBF Interpolator Class ---
class RBFInterpolator:
    def __init__(self, points_xy, values_z, rbf_type='gaussian', epsilon=1.0, lambda_reg=0.0):
        """
        Initializes and fits the RBF interpolator.

        Args:
            points_xy (np.ndarray): Nx2 array of (x, y) coordinates of data points.
            values_z (np.ndarray): N array of corresponding z values.
            rbf_type (str): Type of RBF to use ('gaussian', 'multiquadric', etc.).
            epsilon (float): Shape parameter for RBFs that require it.
            lambda_reg (float): Tikhonov regularization parameter.
        """
        if rbf_type not in RBF_FUNCTIONS:
            raise ValueError(f"Unsupported RBF type: {rbf_type}. Supported: {list(RBF_FUNCTIONS.keys())}")

        self.points_xy = np.asarray(points_xy)
        self.values_z = np.asarray(values_z)
        self.rbf_func = RBF_FUNCTIONS[rbf_type]
        self.epsilon = epsilon
        self.lambda_reg = lambda_reg
        self.weights = None
        
        self.N = self.points_xy.shape[0]
        if self.N != self.values_z.shape[0]:
            raise ValueError("Number of points (xy) must match number of values (z).")

        self._solve_weights()

    def _solve_weights(self):
        # Construct the interpolation matrix Phi
        phi_matrix = np.zeros((self.N, self.N), dtype=np.float64)
        for i in range(self.N):
            for j in range(self.N):
                # Euclidean distance between point i and point j
                dist = np.linalg.norm(self.points_xy[i] - self.points_xy[j])
                phi_matrix[i, j] = self.rbf_func(dist, self.epsilon)
        
        # Apply Tikhonov regularization if lambda_reg > 0
        if self.lambda_reg > 0:
            phi_matrix += self.lambda_reg * np.eye(self.N)
            
        # Solve the linear system Phi * w = z
        try:
            # self.weights = np.linalg.solve(phi_matrix, self.values_z)
            self.weights = scipy.linalg.solve(phi_matrix, self.values_z) # Often more robust
        except np.linalg.LinAlgError as e:
            print(f"Linear algebra error during solving for weights: {e}")
            print("This might be due to a singular or ill-conditioned matrix.")
            print("Try adjusting epsilon, lambda_reg, or using a different RBF type.")
            self.weights = None # Indicate failure
        except Exception as e_gen:
            print(f"An unexpected error occurred during solving for weights: {e_gen}")
            self.weights = None


    def interpolate(self, query_points_xy):
        """
        Interpolates z values at new query_points_xy.

        Args:
            query_points_xy (np.ndarray): Mx2 array of (x, y) coordinates where to interpolate.

        Returns:
            np.ndarray: M array of interpolated z values, or None if weights were not solved.
        """
        if self.weights is None:
            print("Error: RBF weights have not been solved. Cannot interpolate.")
            return None
            
        query_points_xy = np.asarray(query_points_xy)
        num_query_points = query_points_xy.shape[0]
        interpolated_values_z = np.zeros(num_query_points, dtype=np.float64)

        for i in range(num_query_points):
            val = 0.0
            for j in range(self.N): # N is number of original data points
                dist = np.linalg.norm(query_points_xy[i] - self.points_xy[j])
                val += self.weights[j] * self.rbf_func(dist, self.epsilon)
            interpolated_values_z[i] = val
            
        return interpolated_values_z

# --- Block 2: RBF Fitting & Visualization ---
def generate_synthetic_data(num_points=100, noise_level=0.1, extent=2*np.pi):
    """Generates noisy samples from z = sin(x) * cos(y)."""
    # Scattered points for interpolation
    x_scattered = np.random.uniform(-extent/2, extent/2, num_points)
    y_scattered = np.random.uniform(-extent/2, extent/2, num_points)
    
    z_true_scattered = np.sin(x_scattered) * np.cos(y_scattered)
    noise = np.random.normal(0, noise_level, num_points)
    z_noisy_scattered = z_true_scattered + noise
    
    points_xy = np.vstack((x_scattered, y_scattered)).T
    
    # Grid for plotting ground truth and interpolated surface
    x_grid_range = np.linspace(-extent/2, extent/2, 50)
    y_grid_range = np.linspace(-extent/2, extent/2, 50)
    X_grid, Y_grid = np.meshgrid(x_grid_range, y_grid_range)
    Z_true_grid = np.sin(X_grid) * np.cos(Y_grid)
    
    query_points_xy_grid = np.vstack((X_grid.ravel(), Y_grid.ravel())).T
    
    return points_xy, z_noisy_scattered, X_grid, Y_grid, Z_true_grid, query_points_xy_grid

def plot_rbf_fit(X_grid, Y_grid, Z_true_grid, points_xy_scattered, z_noisy_scattered, Z_interpolated_grid,
                 title="RBF Interpolation Fit", save_filename=None):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot ground truth surface
    ax.plot_surface(X_grid, Y_grid, Z_true_grid, alpha=0.3, cmap='viridis', label='Ground Truth')
    
    # Plot noisy scattered data points
    ax.scatter(points_xy_scattered[:,0], points_xy_scattered[:,1], z_noisy_scattered, 
               c='red', marker='o', s=20, label='Noisy Samples')
               
    # Plot RBF interpolated surface
    if Z_interpolated_grid is not None:
        ax.plot_surface(X_grid, Y_grid, Z_interpolated_grid.reshape(X_grid.shape), 
                        alpha=0.7, cmap='plasma', label='RBF Fit')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    # ax.legend() # Legend can be messy with surfaces

    if save_filename:
        plt.savefig(os.path.join(OUTPUT_DIR, save_filename))
        print(f"Plot saved to {os.path.join(OUTPUT_DIR, save_filename)}")
    plt.show()


if __name__ == "__main__":
    print("--- Generating Synthetic Data ---")
    num_data_points = 50
    noise = 0.2
    points_xy_data, z_data_noisy, X_grid_plot, Y_grid_plot, Z_true_plot, query_points_grid_plot = \
        generate_synthetic_data(num_points=num_data_points, noise_level=noise)

    print("\n--- Testing RBF Interpolator ---")

    # Experiment 1: Gaussian RBF, varying epsilon (shape parameter)
    print("\nExperiment 1: Gaussian RBF, varying epsilon")
    epsilons_gauss = [0.5, 1.0, 2.0] # Epsilon for Gaussian, smaller is "spikier"
    for eps_g in epsilons_gauss:
        title_g = f"Gaussian RBF (epsilon={eps_g:.1f}, lambda=0)"
        try:
            rbf_interp_gauss = RBFInterpolator(points_xy_data, z_data_noisy, 
                                               rbf_type='gaussian', epsilon=eps_g, lambda_reg=0)
            if rbf_interp_gauss.weights is not None:
                Z_interp_gauss = rbf_interp_gauss.interpolate(query_points_grid_plot)
                plot_rbf_fit(X_grid_plot, Y_grid_plot, Z_true_plot, points_xy_data, z_data_noisy, Z_interp_gauss,
                             title=title_g, save_filename=f"rbf_gauss_eps{str(eps_g).replace('.','')}.png")
            else:
                print(f"  Skipping plot for Gaussian RBF (epsilon={eps_g}) due to weight solving error.")
        except Exception as e:
            print(f"  Error with Gaussian RBF (epsilon={eps_g}): {e}")


    # Experiment 2: Multiquadric RBF, varying epsilon
    print("\nExperiment 2: Multiquadric RBF, varying epsilon")
    epsilons_mq = [0.5, 1.0, 2.0] # Epsilon for Multiquadric
    for eps_m in epsilons_mq:
        title_m = f"Multiquadric RBF (epsilon={eps_m:.1f}, lambda=0)"
        try:
            rbf_interp_mq = RBFInterpolator(points_xy_data, z_data_noisy, 
                                            rbf_type='multiquadric', epsilon=eps_m, lambda_reg=0)
            if rbf_interp_mq.weights is not None:
                Z_interp_mq = rbf_interp_mq.interpolate(query_points_grid_plot)
                plot_rbf_fit(X_grid_plot, Y_grid_plot, Z_true_plot, points_xy_data, z_data_noisy, Z_interp_mq,
                             title=title_m, save_filename=f"rbf_mq_eps{str(eps_m).replace('.','')}.png")
            else:
                print(f"  Skipping plot for Multiquadric RBF (epsilon={eps_m}) due to weight solving error.")
        except Exception as e:
            print(f"  Error with Multiquadric RBF (epsilon={eps_m}): {e}")
            
    # Experiment 3: Gaussian RBF with varying regularization (lambda)
    print("\nExperiment 3: Gaussian RBF (epsilon=1.0), varying lambda_reg")
    epsilon_fixed_gauss = 1.0
    lambdas = [0, 1e-4, 1e-2, 1e-1]
    for lam in lambdas:
        title_l = f"Gaussian RBF (eps={epsilon_fixed_gauss:.1f}, lambda={lam:.0e})"
        try:
            rbf_interp_reg = RBFInterpolator(points_xy_data, z_data_noisy, 
                                             rbf_type='gaussian', epsilon=epsilon_fixed_gauss, lambda_reg=lam)
            if rbf_interp_reg.weights is not None:
                Z_interp_reg = rbf_interp_reg.interpolate(query_points_grid_plot)
                plot_rbf_fit(X_grid_plot, Y_grid_plot, Z_true_plot, points_xy_data, z_data_noisy, Z_interp_reg,
                             title=title_l, save_filename=f"rbf_gauss_lambda{str(lam).replace('.','p').replace('-','m')}.png")
            else:
                print(f"  Skipping plot for Gaussian RBF (lambda={lam}) due to weight solving error.")
        except Exception as e:
            print(f"  Error with Gaussian RBF (lambda={lam}): {e}")

    # Experiment 4: Thin-Plate Spline (no epsilon, try with/without regularization)
    print("\nExperiment 4: Thin-Plate Spline RBF")
    lambdas_tps = [0, 1e-3] # TPS can be sensitive, try small regularization
    for lam_tps in lambdas_tps:
        title_tps = f"Thin-Plate Spline RBF (lambda={lam_tps:.0e})"
        try:
            rbf_interp_tps = RBFInterpolator(points_xy_data, z_data_noisy,
                                             rbf_type='thin_plate_spline', lambda_reg=lam_tps)
            if rbf_interp_tps.weights is not None:
                Z_interp_tps = rbf_interp_tps.interpolate(query_points_grid_plot)
                plot_rbf_fit(X_grid_plot, Y_grid_plot, Z_true_plot, points_xy_data, z_data_noisy, Z_interp_tps,
                             title=title_tps, save_filename=f"rbf_tps_lambda{str(lam_tps).replace('.','p').replace('-','m')}.png")
            else:
                print(f"  Skipping plot for TPS RBF (lambda={lam_tps}) due to weight solving error.")
        except Exception as e:
            print(f"  Error with TPS RBF (lambda={lam_tps}): {e}")
            
    print(f"\nAll experiments complete. Check plots and saved images in '{OUTPUT_DIR}'.")