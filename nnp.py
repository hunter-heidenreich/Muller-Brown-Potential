import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# --- 1. Ground Truth (Muller-Brown Class) ---
class MuellerBrownGT(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("A", torch.tensor([-200., -100., -170., 15.]))
        self.register_buffer("a", torch.tensor([-1., -1., -6.5, 0.7]))
        self.register_buffer("b", torch.tensor([0., 0., 11., 0.6]))
        self.register_buffer("c", torch.tensor([-10., -10., -6.5, 0.7]))
        self.register_buffer("x0", torch.tensor([1., 0., -0.5, -1.]))
        self.register_buffer("y0", torch.tensor([0., 0.5, 1.5, 1.]))

    def forward(self, coords):
        # coords shape: (N, 2)
        x, y = coords[:, 0], coords[:, 1]
        V = torch.zeros_like(x)
        for i in range(4):
            dx = x - self.x0[i]
            dy = y - self.y0[i]
            V += self.A[i] * torch.exp(self.a[i]*dx**2 + self.b[i]*dx*dy + self.c[i]*dy**2)
        return V.unsqueeze(1) # Shape (N, 1)

# --- 2. The Student Model (Simple MLP) ---
class SimpleNNP(nn.Module):
    def __init__(self):
        super().__init__()
        # A simple 3-layer MLP with Tanh activation (good for smooth surfaces)
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.net(x)

# --- 3. Setup Data and Training ---
torch.manual_seed(42)
gt_model = MuellerBrownGT()
student_model = SimpleNNP()

# Generate Training Data (Random sampling over the relevant domain)
n_samples = 2000
X_train = (torch.rand(n_samples, 2) * torch.tensor([2.7, 2.5])) + torch.tensor([-1.5, -0.5])
y_train = gt_model(X_train)

optimizer = optim.Adam(student_model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Grid for visualization
res = 100
x_grid = np.linspace(-1.5, 1.2, res)
y_grid = np.linspace(-0.5, 2.0, res)
X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
X_flat = torch.tensor(np.stack([X_mesh.flatten(), Y_mesh.flatten()], axis=1), dtype=torch.float32)

# Get Ground Truth Surface
with torch.no_grad():
    Z_gt = gt_model(X_flat).numpy().reshape(res, res)
    
# --- 4. Training Snapshots ---
snapshots = []
epochs = [0, 50, 1000] # Capture untrained, early, and late stages

print("Training Neural Network Potential...")
for epoch in range(epochs[-1] + 1):
    optimizer.zero_grad()
    outputs = student_model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if epoch in epochs:
        print(f"Snapshot at epoch {epoch}, Loss: {loss.item():.4f}")
        student_model.eval()
        with torch.no_grad():
            Z_pred = student_model(X_flat).numpy().reshape(res, res)
            snapshots.append(Z_pred)
        student_model.train()

# --- 5. Plotting ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
levels = np.linspace(-160, 120, 40) # Fixed levels for consistent comparison
cmap = 'viridis'

titles = [
    "Ground Truth\n(Müller-Brown Analytical)",
    f"Neural Network (Epoch {epochs[1]})\nEarly Training - High Error",
    f"Neural Network (Epoch {epochs[2]})\nConverged - Learning Complete"
]

data_to_plot = [Z_gt, snapshots[1], snapshots[2]]

for ax, data, title in zip(axes, data_to_plot, titles):
    # Clip data for cleaner visualization so extremely high predictions don't wash out colors
    data_clipped = np.clip(data, -160, 120)
    contour = ax.contourf(X_mesh, Y_mesh, data_clipped, levels=levels, cmap=cmap, extend='both')
    ax.contour(X_mesh, Y_mesh, Z_gt, levels=levels[::4], colors='k', linewidths=0.5, alpha=0.3) # Overlay GT lines for reference
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("X")
    if ax == axes[0]:
        ax.set_ylabel("Y")

# Add a colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(contour, cax=cbar_ax, label='Potential Energy')

plt.suptitle("Benchmarking Machine Learning: Learning the Müller-Brown Surface", fontsize=16, fontweight='bold', y=1.05)
plt.tight_layout(rect=[0, 0, 0.9, 1]) # Make room for colorbar and title
plt.savefig('muller_brown_ml_benchmark.png', dpi=150, bbox_inches='tight')
plt.show()