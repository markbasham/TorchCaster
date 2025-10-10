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

# --- Define two 2D planes (start and end) ---
# Start plane at z=0, End plane at z=D-1
# We'll make a square grid across y and x

po = 32               # perspective offset for the simple setup

# Define start and end corners
start_top_left  = np.array([0, po ,  po], dtype=np.float32)
start_top_right = np.array([0,  H-po, po], dtype=np.float32)
start_bottom_left  = np.array([0, po, W-po], dtype=np.float32)
start_bottom_right = np.array([0, H-po, W-po], dtype=np.float32)

end_top_left  = np.array([D-1, 0, 0], dtype=np.float32)
end_top_right = np.array([D-1, H-1, 0], dtype=np.float32)
end_bottom_left  = np.array([D-1, 0, W-1], dtype=np.float32)
end_bottom_right = np.array([D-1, H-1, W-1], dtype=np.float32)


m_y, m_x = 512, 512   # number of rays in y and x directions
n = 256               # number of samples per ray


# --- Interpolate start and end planes ---
# Each will be shape (m_y, m_x, 3)
def bilinear_grid(top_left, top_right, bottom_left, bottom_right, m_y, m_x):
    v = torch.linspace(0, 1, m_y).view(m_y, 1, 1)
    u = torch.linspace(0, 1, m_x).view(1, m_x, 1)
    top = torch.tensor(top_left) + (torch.tensor(top_right) - torch.tensor(top_left)) * u
    bottom = torch.tensor(bottom_left) + (torch.tensor(bottom_right) - torch.tensor(bottom_left)) * u
    return top + (bottom - top) * v

start_plane = bilinear_grid(start_top_left, start_top_right, start_bottom_left, start_bottom_right, m_y, m_x)
end_plane   = bilinear_grid(end_top_left,   end_top_right,   end_bottom_left,   end_bottom_right,   m_y, m_x)

start_plane.to(device)
end_plane.to(device)

# --- Build rays and sample points ---
t = torch.linspace(0, 1, n).view(1, 1, n, 1)  # (1,1,n,1)
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

plt.figure(figsize=(20,20))
plt.imshow(projection, cmap='viridis', origin='lower')
plt.colorbar(label="Sampled Intensity (mid-depth)")
plt.title("2D Grid of Rays Sampling Mid-Depth Values")
plt.xlabel("x-ray index")
plt.ylabel("y-ray index")
plt.tight_layout()
plt.savefig("samples_grid_middepth.png", dpi=200)
print("Saved 2D ray-grid visualization to samples_grid_middepth.png")

from PIL import Image

# projection is your 2D NumPy array, e.g. shape (m_y, m_x)
# First, normalize to 0–255 range for 8-bit image output
proj = projection - projection.min()
proj = proj / proj.max()
proj = (proj * 255).astype(np.uint8)

# Create and save grayscale image
#img = Image.fromarray(proj, mode="L")  # "L" = 8-bit grayscale
img = Image.fromarray(np.stack([proj]*3, axis=-1), mode="RGB")
img.save("projection_direct.png")

print("Saved grayscale projection image to projection_direct.png")

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
img.save("projection_direct_rgb.png")

print("Saved grayscale projection image to projection_direct_rgb.png")