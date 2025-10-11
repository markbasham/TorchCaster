import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Select your target GPU (0-indexed)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- Create a 3D volume with k random Gaussians ---
D, H, W = 256, 256, 256
k = 10  # number of random Gaussians

z, y, x = np.meshgrid(
    np.linspace(-1, 1, D),
    np.linspace(-1, 1, H),
    np.linspace(-1, 1, W),
    indexing='ij'
)

volume = np.zeros((D, H, W), dtype=np.float32)

for i in range(k):
    # Random center in normalized coordinates (-1 to 1)
    cx, cy, cz = np.random.uniform(-0.8, 0.8, 3)
    
    # Random width (smaller = sharper Gaussian)
    sigma = np.random.uniform(0.15, 0.4)
    
    # Random amplitude (brightness)
    amplitude = np.random.uniform(0.5, 1.0)
    
    # Compute Gaussian and add it to the volume
    g = amplitude * np.exp(-((x - cx)**2 + (y - cy)**2 + (z - cz)**2) / (2 * sigma**2))
    volume += g.astype(np.float32)

vol_t = torch.from_numpy(volume)[None, None].to(device)  # (1, 1, D, H, W)

print(f"Generated volume with {k} random Gaussians.")

# Now sort out the camera
def generate_camera_planes(
    res_y, res_x,
    fov=60.0,                     # field of view in degrees
    aspect=1.0,                   # width/height ratio
    distance=64.0,                # distance to far plane (volume depth)
    perspective=1.0,              # 0 = orthographic, 1 = perspective
    cam_origin=(0, 0, -32),       # camera position
    cam_forward=(0, 0, 1),        # camera viewing direction
    cam_up=(0, 1, 0),             # up vector
    device="cpu"
):
    """
    Generates start and end planes for ray marching through a 3D volume,
    using a simple pinhole-style camera model.
    """
    cam_origin = torch.tensor(cam_origin, dtype=torch.float32, device=device)
    forward = torch.tensor(cam_forward, dtype=torch.float32, device=device)
    up = torch.tensor(cam_up, dtype=torch.float32, device=device)

    # Normalize orientation vectors
    forward = forward / torch.norm(forward)
    right = torch.cross(forward, up)
    right = right / torch.norm(right)
    up = torch.cross(right, forward)
    up = up / torch.norm(up)

    # Image plane setup
    fov_rad = np.deg2rad(fov)
    half_height = torch.tan(torch.tensor(fov_rad / 2, device=device))
    half_width = aspect * half_height

    # Create 2D grid of pixel coordinates in NDC [-1, 1]
    v = torch.linspace(-1, 1, res_y, device=device).view(res_y, 1)
    u = torch.linspace(-1, 1, res_x, device=device).view(1, res_x)
    uu, vv = torch.meshgrid(u[0], v[:, 0], indexing='xy')  # (res_y, res_x)

    # Project to camera space
    ray_dir = (
        forward[None, None, :] +
        uu[..., None] * half_width * right[None, None, :] +
        vv[..., None] * half_height * up[None, None, :]
    )
    ray_dir = ray_dir / torch.norm(ray_dir, dim=-1, keepdim=True)

    # Start plane = near plane at camera origin
    start_plane = cam_origin[None, None, :].expand(res_y, res_x, 3)

    # End plane = point along ray direction at given distance
    end_plane = start_plane + ray_dir * distance

    # Perspective control (lerp toward parallel rays if perspective < 1)
    if perspective < 1.0:
        # Orthographic equivalent direction (just camera forward)
        ortho_end = start_plane + forward[None, None, :] * distance
        end_plane = torch.lerp(ortho_end, end_plane, perspective)

    return start_plane, end_plane

m_y, m_x = 512, 512   # number of rays in y and x directions
n = 256               # number of samples per ray

# Add an animation loop
for i in range(0,100, 5):

    start_plane, end_plane = generate_camera_planes(
        res_y=m_y,
        res_x=m_x,
        fov=256.0,              # degrees
        aspect=m_x / m_y,
        distance=150.0,         # matches your volume depth
        perspective=1.0,       # 1.0 = full perspective, 0.0 = orthographic
        cam_origin=(128, 128, i), # Set to move forward through the volume
        cam_forward=(0, 0, 1),
        cam_up=(0, 1, 0),
        device=device
    )

    # --- Build rays and sample points ---
    t = torch.linspace(0, 1, n).view(1, 1, n, 1).to(device)  # (1,1,n,1)
    starts = start_plane[:, :, None, :]  # (m_y, m_x, 1, 3)
    ends   = end_plane[:, :, None, :]    # (m_y, m_x, 1, 3)
    coords = starts + (ends - starts) * t  # (m_y, m_x, n, 3)

    # --- Normalize to [-1, 1] ---
    coords_norm = torch.zeros_like(coords).to(device)
    coords_norm[..., 0] = 2.0 * coords[..., 2] / (W - 1) - 1.0  # x
    coords_norm[..., 1] = 2.0 * coords[..., 1] / (H - 1) - 1.0  # y
    coords_norm[..., 2] = 2.0 * coords[..., 0] / (D - 1) - 1.0  # z

    # --- Reshape to feed grid_sample ---
    # Combine the m_y * m_x rays into one batch
    M = m_y * m_x
    grid = coords_norm.view(M, 1, 1, n, 3).to(device)
    print('grid shape', grid.shape)
    samples = F.grid_sample(
        vol_t.expand(M, -1, -1, -1, -1),
        grid,
        mode='bilinear',
        align_corners=True,
    )
    print('samples shape', samples.shape)
    samples = samples.view(m_y, m_x, n)

    print('samples shape', samples.shape)

    # --- Normalize to [0, 1] for colormap lookup ---
    samples_min, samples_max = samples.min(), samples.max()
    samples_norm = (samples - samples_min) / (samples_max - samples_min + 1e-8)

    # --- Create viridis colormap as a tensor ---
    viridis = matplotlib.cm.get_cmap('viridis', 256)
    cmap_array = torch.tensor(viridis(np.linspace(0, 1, 256)), dtype=torch.float32, device=device)  # (256, 4)

    # --- Map normalized samples to RGBA using lookup ---
    idx = (samples_norm * 255).long().clamp(0, 255)
    rgba = cmap_array[idx]  # (x, y, n, 4)

    # --- Replace the alpha channel with the raw (normalized) intensity ---
    rgba[..., 3] = samples

    samples_rgba = rgba.detach().cpu().numpy()  # shape (x, y, n, 4)
    print("Sample volume RGBA shape:", samples_rgba.shape)  # (m_y, m_x, n)

    samples = samples.detach().cpu().numpy()
    print("Sample volume shape:", samples.shape)  # (m_y, m_x, n)

    # --- Save a visualization ---
    # Make a projection
    projection = samples.sum(axis=-1)  # shape (m_y, m_x)

    from PIL import Image

    # --- Save an RGBA visualization ---
    # Make a projection
    projection = samples_rgba.sum(axis=2)  # shape (m_y, m_x)

    # projection is 3d NumPy array, e.g. shape (m_y, m_x)
    # First, normalize to 0–255 range for 8-bit image output
    proj = projection - projection.min()
    proj = proj / proj.max()
    proj = (proj * 255).astype(np.uint8)

    # Create and save colour image
    #img = Image.fromarray(proj, mode="L")  # "L" = 8-bit grayscale
    img = Image.fromarray(proj, mode="RGBA")
    img.save(f"projection_direct_rgb_{i:02d}.png")

    print("Saved Colour projection image to projection_direct_rgb.png")