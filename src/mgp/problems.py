from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .newton import NewtonResult, solve_nonlinear_system


def _as_parameter_tensor(mu) -> torch.Tensor:
    if isinstance(mu, torch.Tensor):
        return mu.to(dtype=torch.float64)
    return torch.tensor(mu, dtype=torch.float64)


def _trapezoid(values: np.ndarray, *, dx: float, axis: int = -1) -> np.ndarray:
    """NumPy-version-compatible trapezoidal integration helper."""
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    return integrate(values, dx=dx, axis=axis)


@dataclass
class NonlinearDiffusionProblem:
    num_nodes: int = 65
    diffusion_nonlinearity: float = 12.0
    reaction_strength: float = 0.5

    def __post_init__(self) -> None:
        if self.num_nodes < 3:
            raise ValueError("num_nodes must be at least 3")
        self.nodes = torch.linspace(0.0, 1.0, self.num_nodes, dtype=torch.float64)
        self.midpoints = 0.5 * (self.nodes[:-1] + self.nodes[1:])
        self.h = float(self.nodes[1] - self.nodes[0])
        self.interior_dim = self.num_nodes - 2
        self.qoi_name = "state_integral"

    def source(self, mu) -> torch.Tensor:
        parameters = _as_parameter_tensor(mu)
        if parameters.numel() != 3:
            raise ValueError("mu must contain [amplitude, center, width]")
        amplitude, center, width = parameters
        width = torch.clamp(width, min=0.04)
        gaussian = amplitude * torch.exp(-0.5 * ((self.midpoints - center) / width) ** 2)
        sinusoid = 0.2 * amplitude * torch.sin(2.0 * torch.pi * self.midpoints)
        return gaussian + sinusoid

    def residual_torch(self, u_interior: torch.Tensor, mu) -> torch.Tensor:
        u = torch.zeros(self.num_nodes, dtype=torch.float64, device=u_interior.device)
        u[1:-1] = u_interior
        residual = torch.zeros_like(u)
        source_values = self.source(mu).to(u_interior.device)

        for element in range(self.num_nodes - 1):
            ua = u[element]
            ub = u[element + 1]
            gradient = (ub - ua) / self.h
            midpoint_state = 0.5 * (ua + ub)
            conductivity = 1.0 + self.diffusion_nonlinearity * midpoint_state**2
            diffusion_local = conductivity * gradient * torch.tensor(
                [-1.0, 1.0], dtype=torch.float64, device=u_interior.device
            )
            midpoint_residual = self.reaction_strength * midpoint_state**3 - source_values[element]
            reaction_local = midpoint_residual * self.h * 0.5 * torch.tensor(
                [1.0, 1.0], dtype=torch.float64, device=u_interior.device
            )
            residual[element : element + 2] += diffusion_local + reaction_local

        return residual[1:-1]

    def solve_full(self, mu, *, initial: np.ndarray | None = None) -> NewtonResult:
        if initial is None:
            x0 = torch.zeros(self.interior_dim, dtype=torch.float64)
        else:
            x0 = torch.tensor(initial, dtype=torch.float64)
        return solve_nonlinear_system(lambda x: self.residual_torch(x, mu), x0)

    def sample_parameters(self, num_samples: int, *, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        amplitudes = rng.uniform(0.5, 1.5, size=num_samples)
        centers = rng.uniform(0.2, 0.8, size=num_samples)
        widths = rng.uniform(0.07, 0.18, size=num_samples)
        return np.column_stack([amplitudes, centers, widths])

    def sample_parameters_ood(self, num_samples: int, *, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        amplitudes = rng.uniform(1.5, 1.9, size=num_samples)
        centers = np.where(
            rng.random(num_samples) < 0.5,
            rng.uniform(0.1, 0.18, size=num_samples),
            rng.uniform(0.82, 0.9, size=num_samples),
        )
        widths = rng.uniform(0.04, 0.07, size=num_samples)
        return np.column_stack([amplitudes, centers, widths])

    def quantity_of_interest(self, u_interior: np.ndarray) -> float:
        full_state = np.zeros(self.num_nodes, dtype=np.float64)
        full_state[1:-1] = u_interior
        return float(_trapezoid(full_state, dx=self.h))

    def qoi_error(self, reference: float, candidate: float) -> float:
        return float(abs(reference - candidate) / max(abs(reference), 1e-12))

    def parameter_grid(
        self,
        amplitudes: tuple[float, ...],
        centers: tuple[float, ...],
        widths: tuple[float, ...],
    ) -> np.ndarray:
        points = []
        for amplitude in amplitudes:
            for center in centers:
                for width in widths:
                    points.append((amplitude, center, width))
        return np.asarray(points, dtype=np.float64)


@dataclass
class FrontLayerProblem(NonlinearDiffusionProblem):
    diffusion_nonlinearity: float = 20.0
    reaction_strength: float = 1.1
    qoi_variant: str = "sign_crossing"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.qoi_variant == "sign_crossing":
            self.qoi_name = "front_location"
        elif self.qoi_variant == "midrange":
            self.qoi_name = "front_location_midrange"
        else:
            raise ValueError(
                "qoi_variant must be 'sign_crossing' or 'midrange', "
                f"got {self.qoi_variant!r}"
            )

    def source(self, mu) -> torch.Tensor:
        parameters = _as_parameter_tensor(mu)
        if parameters.numel() != 3:
            raise ValueError("mu must contain [amplitude, center, width]")
        amplitude, center, width = parameters
        width = torch.clamp(width, min=0.035, max=0.12)
        left_center = torch.clamp(center - 0.7 * width, min=0.1, max=0.9)
        right_center = torch.clamp(center + 0.7 * width, min=0.1, max=0.9)
        layer_width = torch.clamp(0.45 * width, min=0.02)

        left_peak = amplitude * torch.exp(-0.5 * ((self.midpoints - left_center) / layer_width) ** 2)
        right_peak = amplitude * torch.exp(
            -0.5 * ((self.midpoints - right_center) / layer_width) ** 2
        )
        interface = 0.35 * amplitude * torch.tanh((self.midpoints - center) / layer_width)
        ripple = 0.05 * amplitude * torch.sin(4.0 * torch.pi * self.midpoints)
        return left_peak - right_peak + interface + ripple

    def sample_parameters(self, num_samples: int, *, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        amplitudes = rng.uniform(0.8, 1.8, size=num_samples)
        centers = rng.uniform(0.25, 0.75, size=num_samples)
        widths = rng.uniform(0.04, 0.09, size=num_samples)
        return np.column_stack([amplitudes, centers, widths])

    def sample_parameters_ood(self, num_samples: int, *, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        amplitudes = rng.uniform(1.8, 2.2, size=num_samples)
        centers = np.where(
            rng.random(num_samples) < 0.5,
            rng.uniform(0.15, 0.23, size=num_samples),
            rng.uniform(0.77, 0.85, size=num_samples),
        )
        widths = rng.uniform(0.03, 0.04, size=num_samples)
        return np.column_stack([amplitudes, centers, widths])

    def quantity_of_interest(self, u_interior: np.ndarray) -> float:
        if self.qoi_variant == "midrange":
            return self.front_location_midrange(u_interior)
        full_state = np.zeros(self.num_nodes, dtype=np.float64)
        full_state[1:-1] = u_interior
        signs = np.sign(full_state)
        crossings = np.where(signs[:-1] * signs[1:] < 0.0)[0]
        if len(crossings) == 0:
            return float(self.nodes[int(np.argmin(np.abs(full_state)))])
        index = int(crossings[0])
        ua = full_state[index]
        ub = full_state[index + 1]
        xa = float(self.nodes[index])
        xb = float(self.nodes[index + 1])
        if abs(ub - ua) < 1e-12:
            return xa
        return float(xa - ua * (xb - xa) / (ub - ua))

    def front_location_midrange(self, u_interior: np.ndarray) -> float:
        """Locate the crossing of the state's own midrange level.

        The search is restricted to the segment joining the nodes where the
        full state reaches its minimum and maximum.  Unlike the published
        sign-crossing surrogate, this sensitivity metric remains
        discriminative for one-signed front-layer states.
        """
        full_state = np.zeros(self.num_nodes, dtype=np.float64)
        full_state[1:-1] = u_interior
        index_min = int(np.argmin(full_state))
        index_max = int(np.argmax(full_state))
        lo = min(index_min, index_max)
        hi = max(index_min, index_max)
        if hi - lo < 1:
            return float(self.nodes[int(np.argmin(np.abs(full_state)))])
        level = 0.5 * (full_state[index_min] + full_state[index_max])
        segment = full_state[lo : hi + 1] - level
        signs = np.sign(segment)
        crossings = np.where(signs[:-1] * signs[1:] < 0.0)[0]
        if len(crossings) == 0:
            return float(self.nodes[lo + int(np.argmin(np.abs(segment)))])
        index = int(crossings[0])
        za = segment[index]
        zb = segment[index + 1]
        xa = float(self.nodes[lo + index])
        xb = float(self.nodes[lo + index + 1])
        if abs(zb - za) < 1e-12:
            return xa
        return float(xa - za * (xb - xa) / (zb - za))

    def qoi_error(self, reference: float, candidate: float) -> float:
        return float(abs(reference - candidate))


@dataclass
class BratuSourceProblem(NonlinearDiffusionProblem):
    reaction_strength: float = 0.0
    diffusion_nonlinearity: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.qoi_name = "peak_state"

    def source(self, mu) -> torch.Tensor:
        parameters = _as_parameter_tensor(mu)
        if parameters.numel() != 3:
            raise ValueError("mu must contain [strength, amplitude, center]")
        _, amplitude, center = parameters
        width = torch.tensor(0.08, dtype=torch.float64, device=self.midpoints.device)
        gaussian = amplitude * torch.exp(-0.5 * ((self.midpoints - center) / width) ** 2)
        bias = 0.15 * amplitude * torch.sin(torch.pi * self.midpoints)
        return gaussian + bias

    def residual_torch(self, u_interior: torch.Tensor, mu) -> torch.Tensor:
        parameters = _as_parameter_tensor(mu).to(u_interior.device)
        if parameters.numel() != 3:
            raise ValueError("mu must contain [strength, amplitude, center]")
        strength = parameters[0]
        u = torch.zeros(self.num_nodes, dtype=torch.float64, device=u_interior.device)
        u[1:-1] = u_interior
        residual = torch.zeros_like(u)
        source_values = self.source(mu).to(u_interior.device)

        for element in range(self.num_nodes - 1):
            ua = u[element]
            ub = u[element + 1]
            gradient = (ub - ua) / self.h
            midpoint_state = 0.5 * (ua + ub)
            diffusion_local = gradient * torch.tensor(
                [-1.0, 1.0], dtype=torch.float64, device=u_interior.device
            )
            midpoint_residual = strength * (torch.exp(midpoint_state) - 1.0) - source_values[element]
            reaction_local = midpoint_residual * self.h * 0.5 * torch.tensor(
                [1.0, 1.0], dtype=torch.float64, device=u_interior.device
            )
            residual[element : element + 2] += diffusion_local + reaction_local

        return residual[1:-1]

    def sample_parameters(self, num_samples: int, *, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        strengths = rng.uniform(1.0, 3.5, size=num_samples)
        amplitudes = rng.uniform(0.3, 1.0, size=num_samples)
        centers = rng.uniform(0.25, 0.75, size=num_samples)
        return np.column_stack([strengths, amplitudes, centers])

    def sample_parameters_ood(self, num_samples: int, *, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        strengths = rng.uniform(3.6, 4.5, size=num_samples)
        amplitudes = rng.uniform(1.0, 1.3, size=num_samples)
        centers = np.where(
            rng.random(num_samples) < 0.5,
            rng.uniform(0.15, 0.22, size=num_samples),
            rng.uniform(0.78, 0.85, size=num_samples),
        )
        return np.column_stack([strengths, amplitudes, centers])

    def quantity_of_interest(self, u_interior: np.ndarray) -> float:
        return float(np.max(u_interior))


@dataclass
class HydrologicConductivityProblem(NonlinearDiffusionProblem):
    diffusion_nonlinearity: float = 8.0
    reaction_strength: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.qoi_name = "outlet_flux"

    def source(self, mu) -> torch.Tensor:
        parameters = _as_parameter_tensor(mu)
        if parameters.numel() != 3:
            raise ValueError("mu must contain [recharge, center, conductivity_scale]")
        recharge, center, conductivity_scale = parameters
        width = torch.tensor(0.09, dtype=torch.float64, device=self.midpoints.device)
        localized_recharge = recharge * torch.exp(-0.5 * ((self.midpoints - center) / width) ** 2)
        background = 0.12 * recharge * (1.0 - self.midpoints)
        drainage = -0.08 * conductivity_scale * torch.ones_like(self.midpoints)
        return localized_recharge + background + drainage

    def residual_torch(self, u_interior: torch.Tensor, mu) -> torch.Tensor:
        parameters = _as_parameter_tensor(mu).to(u_interior.device)
        if parameters.numel() != 3:
            raise ValueError("mu must contain [recharge, center, conductivity_scale]")
        conductivity_scale = parameters[2]
        u = torch.zeros(self.num_nodes, dtype=torch.float64, device=u_interior.device)
        u[1:-1] = u_interior
        residual = torch.zeros_like(u)
        source_values = self.source(mu).to(u_interior.device)

        for element in range(self.num_nodes - 1):
            ua = u[element]
            ub = u[element + 1]
            gradient = (ub - ua) / self.h
            midpoint_head = 0.5 * (ua + ub)
            conductivity = 0.35 + conductivity_scale * torch.exp(0.9 * midpoint_head)
            diffusion_local = conductivity * gradient * torch.tensor(
                [-1.0, 1.0], dtype=torch.float64, device=u_interior.device
            )
            recharge_local = -source_values[element] * self.h * 0.5 * torch.tensor(
                [1.0, 1.0], dtype=torch.float64, device=u_interior.device
            )
            residual[element : element + 2] += diffusion_local + recharge_local

        return residual[1:-1]

    def sample_parameters(self, num_samples: int, *, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        recharge = rng.uniform(0.4, 1.1, size=num_samples)
        centers = rng.uniform(0.2, 0.65, size=num_samples)
        conductivity_scale = rng.uniform(0.55, 1.05, size=num_samples)
        return np.column_stack([recharge, centers, conductivity_scale])

    def sample_parameters_ood(self, num_samples: int, *, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        recharge = rng.uniform(1.1, 1.5, size=num_samples)
        centers = np.where(
            rng.random(num_samples) < 0.5,
            rng.uniform(0.08, 0.18, size=num_samples),
            rng.uniform(0.72, 0.88, size=num_samples),
        )
        conductivity_scale = rng.uniform(1.05, 1.35, size=num_samples)
        return np.column_stack([recharge, centers, conductivity_scale])

    def quantity_of_interest(self, u_interior: np.ndarray) -> float:
        full_state = np.zeros(self.num_nodes, dtype=np.float64)
        full_state[1:-1] = u_interior
        ua = full_state[-2]
        ub = full_state[-1]
        midpoint_head = 0.5 * (ua + ub)
        conductivity = 0.35 + 1.0 * np.exp(0.9 * midpoint_head)
        gradient = (ub - ua) / self.h
        return float(-conductivity * gradient)


@dataclass
class TwoDBurgersProblem(NonlinearDiffusionProblem):
    """Steady scalar two-dimensional Burgers-type benchmark."""

    num_nodes: int = 34
    diffusion_nonlinearity: float = 0.0
    reaction_strength: float = 0.0
    viscosity: float = 0.08
    source_scale: float = 8.0
    source_width: float = 0.12

    def __post_init__(self) -> None:
        if self.num_nodes < 3:
            raise ValueError("num_nodes must be at least 3")
        self.nodes = torch.linspace(0.0, 1.0, self.num_nodes, dtype=torch.float64)
        self.h = float(self.nodes[1] - self.nodes[0])
        self.interior_nodes_per_axis = self.num_nodes - 2
        self.interior_dim = self.interior_nodes_per_axis**2
        interior_nodes = self.nodes[1:-1]
        self.grid_x, self.grid_y = torch.meshgrid(
            interior_nodes, interior_nodes, indexing="ij"
        )
        self.qoi_name = "state_integral"

    def source(self, mu) -> torch.Tensor:
        parameters = _as_parameter_tensor(mu)
        if parameters.numel() != 3:
            raise ValueError("mu must contain [amplitude, center_x, center_y]")
        amplitude, center_x, center_y = parameters
        radius_sq = (self.grid_x - center_x) ** 2 + (self.grid_y - center_y) ** 2
        return self.source_scale * amplitude * torch.exp(
            -0.5 * radius_sq / self.source_width**2
        )

    def residual_torch(self, u_interior: torch.Tensor, mu) -> torch.Tensor:
        u = torch.zeros(
            (self.num_nodes, self.num_nodes),
            dtype=torch.float64,
            device=u_interior.device,
        )
        u[1:-1, 1:-1] = u_interior.reshape(
            self.interior_nodes_per_axis, self.interior_nodes_per_axis
        )
        center = u[1:-1, 1:-1]
        gradient_x = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2.0 * self.h)
        gradient_y = (u[1:-1, 2:] - u[1:-1, :-2]) / (2.0 * self.h)
        laplacian = (
            u[2:, 1:-1]
            + u[:-2, 1:-1]
            + u[1:-1, 2:]
            + u[1:-1, :-2]
            - 4.0 * center
        ) / self.h**2
        source_values = self.source(mu).to(u_interior.device)
        residual = (
            center * (gradient_x + gradient_y)
            - self.viscosity * laplacian
            - source_values
        )
        return residual.reshape(-1) * self.h**2

    def sample_parameters(self, num_samples: int, *, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        amplitudes = rng.uniform(0.8, 1.8, size=num_samples)
        centers_x = rng.uniform(0.3, 0.7, size=num_samples)
        centers_y = rng.uniform(0.3, 0.7, size=num_samples)
        return np.column_stack([amplitudes, centers_x, centers_y])

    def sample_parameters_ood(self, num_samples: int, *, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        amplitudes = rng.uniform(1.8, 2.2, size=num_samples)
        centers_x = np.where(
            rng.random(num_samples) < 0.5,
            rng.uniform(0.15, 0.28, size=num_samples),
            rng.uniform(0.72, 0.85, size=num_samples),
        )
        centers_y = np.where(
            rng.random(num_samples) < 0.5,
            rng.uniform(0.15, 0.28, size=num_samples),
            rng.uniform(0.72, 0.85, size=num_samples),
        )
        return np.column_stack([amplitudes, centers_x, centers_y])

    def quantity_of_interest(self, u_interior: np.ndarray) -> float:
        full_state = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float64)
        full_state[1:-1, 1:-1] = np.asarray(
            u_interior, dtype=np.float64
        ).reshape(self.interior_nodes_per_axis, self.interior_nodes_per_axis)
        row_integrals = _trapezoid(full_state, dx=self.h, axis=1)
        return float(_trapezoid(row_integrals, dx=self.h))

    def parameter_grid(
        self,
        amplitudes: tuple[float, ...],
        centers_x: tuple[float, ...],
        centers_y: tuple[float, ...],
    ) -> np.ndarray:
        points = []
        for amplitude in amplitudes:
            for center_x in centers_x:
                for center_y in centers_y:
                    points.append((amplitude, center_x, center_y))
        return np.asarray(points, dtype=np.float64)


def create_problem(name: str) -> NonlinearDiffusionProblem:
    if name == "nonlinear_diffusion":
        return NonlinearDiffusionProblem()
    if name == "front_layer":
        return FrontLayerProblem()
    if name == "front_layer_midrange_qoi":
        return FrontLayerProblem(qoi_variant="midrange")
    if name == "bratu_source":
        return BratuSourceProblem()
    if name == "hydrologic_conductivity":
        return HydrologicConductivityProblem()
    if name == "two_d_burgers":
        return TwoDBurgersProblem()
    raise ValueError(f"unknown problem: {name}")
