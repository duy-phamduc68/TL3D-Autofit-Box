# Mathematical Process of Fitting 3D Bounding Boxes to 2D Bounding Boxes

This document details the mathematical process and optimization pipeline implemented in the `FinalStage._fit_3d_box()` routine. It explains how a drawn 2D bounding box in a CCTV feed is used to automatically estimate the 3D physical dimensions and world-space location of an object (like a vehicle).

*(Note: The Inverse Perspective Mapping (IPM) functions—`_sat_to_cctv` and `cv2.projectPoints`/`cv2.perspectiveTransform`—are treated as black-box mappings that translate points between the $Z=0$ ground plane in Satellite/Orthographic coordinates and the $2D$ CCTV pixel coordinates.)*

---

## 1. State Variables & Parameters

The goal is to find the optimal 3D bounding box that perfectly fills the user-drawn 2D rectangle in the CCTV view. The optimization space consists of 5 parameters:
$$ \theta = [L, W, H, C_x, C_y] $$

- **$L, W, H$**: The physical Length, Width, and Height of the bounding box (in meters).
- **$C_x, C_y$**: The $(X, Y)$ coordinates of the bounding box's **centroid** projected onto the ground plane ($Z=0$) in the orthographic satellite frame.

There are also several constants initialized from the environment:
- **$\text{cam\_sat}$**: The 2D $(X,Y)$ location of the CCTV camera projected onto the satellite orthographic map.
- **$Z_{cam}$**: The physical height of the CCTV camera from the ground (in meters).
- **$\theta_{heading}$**: The fixed rotational heading (yaw) of the object in the world plane (in degrees).

---

## 2. Initialization & Parallax Correction

When a 2D box is drawn, its pixel center is mapped to the satellite plane using IPM, producing an apparent location on the ground, $P_{app}$. However, since the object has height, the physical centroid of the vehicle does not lie on the ground—it rests at $Z = H/2$. 

Because the camera is elevated ($Z_{cam}$), viewing an elevated point causes it to project further away on the ground plane (parallax). We must compute the true ground center $(C_x, C_y)$ by performing a **Reverse Parallax Correction** on $P_{app}$.

Given initial dimensions $H = 1.55$:
1. Vector from the camera to the apparent point:  
   $$ \vec{v}_{app} = P_{app} - \text{cam\_sat} $$
2. Scale factor based on the centroid height ($H_{cent} = H / 2$):  
   $$ \text{factor} = \frac{Z_{cam}}{Z_{cam} - H_{cent}} $$
3. True centroid on the ground:  
   $$ (C_x, C_y) = \text{cam\_sat} + \frac{\vec{v}_{app}}{\text{factor}} $$

*Before optimization begins, $L, W, H$ are uniformly scaled down by 5% iteratively until the projected 3D box falls entirely inside the drawn 2D rectangle.*

---

## 3. Projection of the 3D Box (Forward Pass)

During optimization, the algorithm repeatedly evaluates the fitness of the parameters. For a given state $[L, W, H, C_x, C_y]$, the 8 corners of the 3D cuboid are projected into the CCTV frame.

### Ground Plane Corners ($Z=0$)
The 4 floor corners are calculated using a 2D rotation matrix:
$$ \begin{bmatrix} X_i \\ Y_i \end{bmatrix} = \begin{bmatrix} \cos(\theta_{heading}) & -\sin(\theta_{heading}) \\ \sin(\theta_{heading}) & \cos(\theta_{heading}) \end{bmatrix} \begin{bmatrix} \pm L/2 \\ \pm W/2 \end{bmatrix} + \begin{bmatrix} C_x \\ C_y \end{bmatrix} $$
Each floor corner $P_{floor,i}$ is passed through the IPM mapper to get CCTV pixel coordinates $(u_i, v_i)$.

### Ceiling Plane Corners ($Z=H$)
To find where the top of the box appears in the camera, the algorithm applies a **Forward Parallax Projection**. The physical ceiling corners exist at $Z=H$. The apparent displacement of these points on the ground plane is proportional to the camera height:

1. Vector from camera to true floor corner:  
   $$ \vec{v}_{true} = P_{floor,i} - \text{cam\_sat} $$
2. Expansion factor for height $H$:  
   $$ \text{factor} = \frac{Z_{cam}}{Z_{cam} - H} $$
3. Apparent projection of the ceiling point on the ground plane:  
   $$ P_{ceil\_app, i} = \text{cam\_sat} + \vec{v}_{true} \cdot \text{factor} $$
   
These $P_{ceil\_app, i}$ coordinates are then passed through the IPM mapper to get the remaining 4 pixel coordinates $(u_{i+4}, v_{i+4})$.

---

## 4. The Loss Function

The algorithm calculates an aggregate Loss based on the 8 projected 2D points. The goal of the optimizer is to minimize this loss.

First, the pixel bounding box of the 8 projected corners is determined:
$$ U_{min} = \min(u_{0..7}), \quad V_{min} = \min(v_{0..7}) $$
$$ U_{max} = \max(u_{0..7}), \quad V_{max} = \max(v_{0..7}) $$

Let the target 2D box drawn by the user have bounds $Rect = [x_{min}, y_{min}, x_{max}, y_{max}]$.

### A. Boundary Penalty ($Pen$)
If `force_touch` is active, the algorithm strictly enforces that the projected 3D box exactly touches the edges of the drawn 2D box using Mean Squared Error (MSE):
$$ Pen = (U_{min} - x_{min})^2 + (V_{min} - y_{min})^2 + (U_{max} - x_{max})^2 + (V_{max} - y_{max})^2 $$

### B. Aspect Ratio Regularization ($R_{pen}$)
To prevent the optimizer from creating physically impossible vehicles (like a 5-meter tall cube), a soft constraint enforces a realistic height-to-width ratio:
$$ R_{pen} = (H - 1.2 \cdot W)^2 $$

### C. Volume Maximization ($Score$)
To encourage the box to fill up the available space robustly (and to prevent it from collapsing to zero if it's already inside the bounds), a linear reward is applied to the dimensions:
$$ Score = L \cdot \omega_L + W \cdot \omega_W + H \cdot \omega_H $$
*(Where $\omega_L, \omega_W, \omega_H$ are user-defined weights).*

### Final Objective
The total loss to minimize is combining these elements:
$$ Loss = -Score + 100000.0 \cdot Pen + 10000.0 \cdot R_{pen} $$

---

## 5. Optimization (Adam)

The parameters are updated over $N$ iterations (default 750) using the Adam Optimizer.

For each iteration, the numerical gradient $\nabla \theta$ is computed using finite differences (epsilon $= 10^{-4}$). The Adam update rule is applied:

1. Update biased first moment estimate:  
   $$ m = \beta_1 \cdot m + (1 - \beta_1) \cdot \nabla \theta $$
2. Update biased second raw moment estimate:  
   $$ v = \beta_2 \cdot v + (1 - \beta_2) \cdot (\nabla \theta)^2 $$
3. Compute bias-corrected moments:  
   $$ \hat{m} = \frac{m}{1 - \beta_1^t}, \quad \hat{v} = \frac{v}{1 - \beta_2^t} $$
4. Update parameters:  
   $$ \theta = \theta - \alpha \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon} $$

*Note: The learning rate $\alpha$ is scaled per parameter to balance physical meters vs. pixel tolerances.*

Finally, hard constraints ensure the object does not shrink below realistic vehicle dimensions:
$$ L \ge 3.0, \quad W \ge 1.5, \quad H \ge 1.55 $$

The parameters resulting in the lowest loss (that respect the bounds) are selected and rendered to the screen.
