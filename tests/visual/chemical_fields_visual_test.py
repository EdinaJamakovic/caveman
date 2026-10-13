import matplotlib

matplotlib.use('TkAgg')

import torch
import matplotlib.pyplot as plt
from src.environment.chemical_fields import ChemicalFields


def run_diffusion_visual_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fields = ChemicalFields(size=32, gamma=(0.999, 0.980, 0.920, 0.900), device=device)

    # Initial drop at center (64, 64)
    center_x = torch.tensor([0.0], device=device)
    center_y = torch.tensor([0.0], device=device)
    amount = torch.tensor([100.0], device=device)

    fields.add_chemical(channel_idx=0, x_coords=center_x, y_coords=center_y, amounts=amount)

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))

    grid_data = fields.grid[0, 0].cpu().numpy()

    # 1. Simple linear scale starting with initial peak concentration
    im = ax.imshow(grid_data, cmap="viridis", vmin=0, vmax=grid_data.max())
    fig.colorbar(im, ax=ax, label="Concentration")
    title = ax.set_title("Step 0")

    for step in range(1, 700):
        fields.diffuse((1,1,1,1),"circular")

        grid_data = fields.grid[0, 0].cpu().numpy()
        peak = grid_data.max()

        # 2. Update image array
        im.set_data(grid_data)

        # 3. Dynamic color scaling: auto-fit vmax to current grid peak
        im.set_clim(vmin=0, vmax=max(peak, 1e-5))

        title.set_text(f"Step {step} | Peak: {peak:.2f}")

        # Render window frame
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.0001)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    run_diffusion_visual_test()