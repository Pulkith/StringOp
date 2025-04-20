# camera intrinsics (pixels), object height (m), gains
f_x, f_y = ... , ...
Z = ...
k_omega, k_v = ... , ...

# image dims
W, H = image.shape[1], image.shape[0]

# bounding box
x1, y1, x2, y2 = bbox  # in pixels

# 1. image‐space quantities
u = 0.5*(x1 + x2)
h = (y2 - y1)

# 2. errors
e_u = u - W/2
theta_err = atan2(e_u, f_x)

d_curr = f_y * Z / h
d_des  = f_y * Z / H
e_d   = d_curr - d_des

# 3. control outputs
omega_z = -k_omega * theta_err
v_x      =  k_v     * e_d

# (optionally clip to your drone’s max v_x and omega_z)