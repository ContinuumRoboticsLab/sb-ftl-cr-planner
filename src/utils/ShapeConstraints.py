import numpy as np


class ConvexShape:
    def __init__(self):
        pass
    
    def get_matplotlib_mesh(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def contains_point(self, point):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    

class cone(ConvexShape):
    def __init__(self, SE3, radius=0.5, height=2, resolution=100):
        self.SE3 = SE3
        self.radius = radius
        self.height = height
        self.resolution = resolution
        
    def get_matplotlib_mesh(self):
        """
        Generate coordinates of a cone transformed by an SE3 pose.
        The cone's TIP will be located at the SE3 translation, oriented along SE3's +Z axis.

        Parameters
        ----------
        SE3 : np.ndarray or object
            Either a 4x4 homogeneous transform or an object with .R (3x3) and .p (3,) attributes.
            The cone tip is positioned at SE3 translation.
        radius : float
            Base radius of the cone.
        height : float
            Height of the cone.
        resolution : int
            Sampling resolution for meshgrid.

        Returns
        -------
        X, Y, Z : np.ndarray
            Transformed meshgrid coordinates for plotting with plot_surface().
        """

        # --- local cone definition ---
        # tip at (0,0,0), base circle at z = height
        u = np.linspace(0, 2 * np.pi, self.resolution)
        v = np.linspace(0, 1, self.resolution)
        u, v = np.meshgrid(u, v)

        x = self.radius * v * np.cos(u)
        y = self.radius * v * np.sin(u)
        z = self.height * v

        pts_local = np.stack([x, y, z], axis=-1).reshape(-1, 3)

        # --- extract rotation & translation ---
        if isinstance(self.SE3, np.ndarray) and self.SE3.shape == (4, 4):
            R = self.SE3[:3, :3]
            t = self.SE3[:3, 3]
        else:
            R = self.SE3.R
            t = self.SE3.p

        # --- apply SE3 transform: world = R * local + t ---
        pts_world = pts_local @ R.T + t

        # reshape back into meshgrid
        X = pts_world[:, 0].reshape(x.shape)
        Y = pts_world[:, 1].reshape(y.shape)
        Z = pts_world[:, 2].reshape(z.shape)

        return X, Y, Z
    
    def contains_point(self, point):
        """
        Check if a point is inside the cone.

        Parameters
        ----------
        point : np.ndarray
            A 3D point (3,).

        Returns
        -------
        bool
            True if the point is inside the cone, False otherwise.
        """
        # Transform point to cone's local frame
        if isinstance(self.SE3, np.ndarray) and self.SE3.shape == (4, 4):
            R = self.SE3[:3, :3]
            t = self.SE3[:3, 3]
        else:
            R = self.SE3.R
            t = self.SE3.p

        point_local = R.T @ (point - t)

        # Check if point is within cone bounds
        z = point_local[2]
        if z < 0 or z > self.height:
            return False

        r_at_z = (z / self.height) * self.radius
        radial_dist = np.linalg.norm(point_local[:2])

        return radial_dist <= r_at_z
