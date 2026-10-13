import torch
import torch.nn.functional as F
from src.environment.chemical_fields import ChemicalFields

def test_atomic_deposition():
    fields = ChemicalFields(size=128, device="cuda" if torch.cuda.is_available() else "cpu")

    # Test 1: Three overlapping drops on the exact same coordinate
    xs = torch.tensor([10.2, 10.8, 10.1], device=fields.device)
    ys = torch.tensor([20.1, 20.4, 20.9], device=fields.device)
    amounts = torch.tensor([0.5, 0.3, 0.2], device=fields.device)

    # All wrap to discrete grid index (10, 20)
    fields.add_chemical(channel_idx=0, x_coords=xs, y_coords=ys, amounts=amounts)

    # Assert atomic addition summed correctly
    deposited_val = fields.grid[0, 0, 20, 10].item()
    assert abs(deposited_val - 1.0) < 1e-5, f"Expected 1.0, got {deposited_val}"
    print("Atomic deposition test passed.")


def test_toroidal_wrapping():
    fields = ChemicalFields(size=128, device="cuda" if torch.cuda.is_available() else "cpu")

    # Test 2: Coordinates beyond grid bounds wrap around
    xs = torch.tensor([128.5, -1.0], device=fields.device)
    ys = torch.tensor([0.0, 0.0], device=fields.device)
    amounts = torch.tensor([1.0, 2.0], device=fields.device)

    fields.add_chemical(channel_idx=0, x_coords=xs, y_coords=ys, amounts=amounts)

    assert fields.grid[0, 0, 0, 0].item() == 1.0, "X=128.5 failed to wrap to X=0"
    assert fields.grid[0, 0, 0, 127].item() == 2.0, "X=-1.0 failed to wrap to X=127"
    print("Toroidal boundary wrapping test passed.")


def test_mass_conservation_during_diffusion():
    fields = ChemicalFields(size=128, gamma = (0.999, 0.980, 0.920, 0.900), device="cuda" if torch.cuda.is_available() else "cpu")

    # Single drop of 100.0 units
    fields.add_chemical(
        channel_idx=0,
        x_coords=torch.tensor([64.0], device=fields.device),
        y_coords=torch.tensor([64.0], device=fields.device),
        amounts=torch.tensor([100.0], device=fields.device)
    )

    initial_mass = fields.grid[0, 0].sum().item()
    fields.diffuse((1,1,1,1),'circular')

    diffused_mass = fields.grid[0, 0].sum().item()
    assert abs(initial_mass - diffused_mass) < 1e-3, f"Mass lost during diffusion! {initial_mass} vs {diffused_mass}"
    print("Mass conservation test passed.")


if __name__ == "__main__":
    test_atomic_deposition()
    test_toroidal_wrapping()
    test_mass_conservation_during_diffusion()