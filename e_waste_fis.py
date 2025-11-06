# e_waste_fis.py
# Sugeno fuzzy inference system for e-waste sorting
# Inputs: Metal Content (0–100), Contamination (0–100), Repairability (0–10), Hazard (0–10)
# Output (Sugeno constant): 25=Refurbish, 50=Material Recovery, 75=Hazardous Handling, 90=Safe Disposal

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple

OUTPUT_CONSTANTS: Dict[str, float] = {
    "Refurbish/Reuse": 25.0,
    "Material Recovery": 50.0,
    "Hazardous Handling": 75.0,
    "Safe Disposal": 90.0,
}


# 1) Membership functions
def tri(x: np.ndarray | float, a: float, b: float, c: float) -> np.ndarray:
    """Triangular MF (a, b, c). Supports shoulder cases when a==b or b==c."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    left = (a < x) & (x < b)
    y[left] = (x[left] - a) / (b - a + 1e-12)
    y[x == b] = 1.0
    right = (b < x) & (x < c)
    y[right] = (c - x[right]) / (c - b + 1e-12)
    y[(x <= a) & (a == b)] = 1.0
    y[(x >= c) & (b == c)] = 1.0
    return np.clip(y, 0.0, 1.0)

def trap(x: np.ndarray | float, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Trapezoidal MF (a, b, c, d). Supports shoulder cases when a==b or c==d."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    left = (a < x) & (x < b)
    y[left] = (x[left] - a) / (b - a + 1e-12)
    y[(b <= x) & (x <= c)] = 1.0
    right = (c < x) & (x < d)
    y[right] = (d - x[right]) / (d - c + 1e-12)
    y[(x <= a) & (a == b)] = 1.0
    y[(x >= d) & (c == d)] = 1.0
    return np.clip(y, 0.0, 1.0)


# 2) Variables & sets
@dataclass
class FuzzyVariable:
    name: str
    u_min: float
    u_max: float
    sets: Dict[str, Tuple[str, Tuple[float, ...]]]  # label -> (mf_type, params)

    def mu(self, label: str, x: float) -> float:
        typ, params = self.sets[label]
        if   typ == "tri":  return float(tri(x, *params))
        elif typ == "trap": return float(trap(x, *params))
        raise ValueError(f"Unknown MF type: {typ}")

# Inputs (exactly as in your 2.2.3)
MetalContent = FuzzyVariable(
    "Metal Content (%)", 0, 100, {
        "Low":    ("trap", (0, 0, 20, 40)),
        "Medium": ("tri",  (30, 50, 70)),
        "High":   ("trap", (60, 80, 100, 100)),
    }
)
Contamination = FuzzyVariable(
    "Contamination (0–100)", 0, 100, {
        "Low":      ("tri", (0, 0, 35)),
        "Moderate": ("tri", (25, 50, 75)),
        "High":     ("tri", (65, 100, 100)),
    }
)
Repairability = FuzzyVariable(
    "Repairability (0–10)", 0, 10, {
        "Poor":     ("tri", (0, 0, 4)),
        "Moderate": ("tri", (3, 5, 7)),
        "Good":     ("tri", (6, 10, 10)),
    }
)
Hazard = FuzzyVariable(
    "Hazard (0–10)", 0, 10, {
        "Safe":      ("trap", (0, 0, 2, 3)),
        "Risky":     ("tri",  (2, 5, 8)),
        "Hazardous": ("trap", (7, 8.5, 10, 10)),
    }
)

# 3) Rule base (13 rules)
@dataclass
class Rule:
    antecedents: List[Tuple[FuzzyVariable, str]]  # e.g., [(Hazard,"Hazardous"), (Contamination,"High")]
    consequent: str                               # label in OUTPUT_CONSTANTS
    def fire(self, inputs: Dict[str, float]) -> float:
        vals = [var.mu(lbl, inputs[var.name]) for var, lbl in self.antecedents]
        return float(min(vals)) if vals else 0.0  # AND=min

RULES: List[Rule] = [
    # Safety-first
    Rule([(Hazard, "Hazardous")], "Hazardous Handling"),
    Rule([(Hazard, "Risky"), (Contamination, "High")], "Hazardous Handling"),
    Rule([(Contamination, "High"), (Repairability, "Poor")], "Safe Disposal"),
    # Value-recovery
    Rule([(MetalContent, "High"), (Contamination, "Low")], "Material Recovery"),
    Rule([(MetalContent, "High"), (Hazard, "Safe")], "Material Recovery"),
    Rule([(MetalContent, "Medium"), (Repairability, "Poor"), (Hazard, "Safe")], "Material Recovery"),
    # Reuse
    Rule([(Repairability, "Good"), (Contamination, "Low"), (Hazard, "Safe")], "Refurbish/Reuse"),
    Rule([(Repairability, "Moderate"), (Contamination, "Low"), (Hazard, "Safe")], "Refurbish/Reuse"),
    # Tie-breakers
    Rule([(Repairability, "Moderate"), (MetalContent, "Medium"), (Contamination, "Moderate")], "Material Recovery"),
    Rule([(Repairability, "Poor"), (MetalContent, "Low"), (Contamination, "Moderate")], "Safe Disposal"),
    Rule([(Hazard, "Risky"), (MetalContent, "High")], "Hazardous Handling"),
    # Contamination-led
    Rule([(Contamination, "High"), (MetalContent, "Medium"), (Hazard, "Safe")], "Material Recovery"),
    Rule([(Contamination, "Moderate"), (Repairability, "Good"), (Hazard, "Risky")], "Hazardous Handling"),
]

# 4) Sugeno inference
def sugeno_infer(metal: float, contam: float, repair: float, hazard: float, return_trace: bool=False):
    """Return (score, nearest_label) or (score, label, trace) if return_trace=True."""
    # Clamp to universes
    metal  = float(np.clip(metal,  MetalContent.u_min,  MetalContent.u_max))
    contam = float(np.clip(contam, Contamination.u_min, Contamination.u_max))
    repair = float(np.clip(repair, Repairability.u_min, Repairability.u_max))
    hazard = float(np.clip(hazard, Hazard.u_min, Hazard.u_max))

    inputs = {
        MetalContent.name:  metal,
        Contamination.name: contam,
        Repairability.name: repair,
        Hazard.name:        hazard,
    }

    weights, outputs, trace = [], [], []
    for rule in RULES:
        w = rule.fire(inputs)
        if w > 0.0:
            z = OUTPUT_CONSTANTS[rule.consequent]
            weights.append(w); outputs.append(z)
            trace.append((rule.consequent, w, z))

    if not weights:
        return (np.nan, "No decision", trace) if return_trace else (np.nan, "No decision")

    w = np.asarray(weights, dtype=float)
    z = np.asarray(outputs, dtype=float)
    score = float(np.sum(w * z) / np.sum(w))
    label = min(OUTPUT_CONSTANTS.keys(), key=lambda k: abs(score - OUTPUT_CONSTANTS[k]))
    return (score, label, trace) if return_trace else (score, label)

# 5) Plotting utilities
def plot_variable(var: FuzzyVariable, path: str):
    xs = np.linspace(var.u_min, var.u_max, 1000)
    plt.figure(figsize=(6, 4))
    for lbl, (typ, params) in var.sets.items():
        y = tri(xs, *params) if typ == "tri" else trap(xs, *params)
        plt.plot(xs, y, label=lbl)
    plt.xlabel(var.name); plt.ylabel("Membership")
    plt.title(f"Membership Functions: {var.name}")
    plt.legend(loc="best"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()

def plot_slice_contamination(metal: float, repair: float, hazard: float, path: str):
    xs = np.linspace(Contamination.u_min, Contamination.u_max, 160)
    ys = [sugeno_infer(metal, c, repair, hazard)[0] for c in xs]
    plt.figure(figsize=(6, 4)); plt.plot(xs, ys)
    plt.xlabel("Contamination (0–100)"); plt.ylabel("Decision Score (0–100)")
    plt.title(f"Decision vs Contamination\n(metal={metal}, repair={repair}, hazard={hazard})")
    plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()

def plot_surface_2d(xvar: FuzzyVariable, yvar: FuzzyVariable, fixed: Dict[str, float],
                    path: str, steps: int = 90):
    x = np.linspace(xvar.u_min, xvar.u_max, steps)
    y = np.linspace(yvar.u_min, yvar.u_max, steps)
    Z = np.zeros((steps, steps), dtype=float)
    for i, xv in enumerate(x):
        for j, yv in enumerate(y):
            vals = {
                MetalContent.name:  fixed.get(MetalContent.name, 50),
                Contamination.name: fixed.get(Contamination.name, 50),
                Repairability.name: fixed.get(Repairability.name, 5),
                Hazard.name:        fixed.get(Hazard.name, 5),
            }
            if xvar is MetalContent:   vals[MetalContent.name]  = xv
            if xvar is Contamination:  vals[Contamination.name] = xv
            if xvar is Repairability:  vals[Repairability.name] = xv
            if xvar is Hazard:         vals[Hazard.name]        = xv
            if yvar is MetalContent:   vals[MetalContent.name]  = yv
            if yvar is Contamination:  vals[Contamination.name] = yv
            if yvar is Repairability:  vals[Repairability.name] = yv
            if yvar is Hazard:         vals[Hazard.name]        = yv
            Z[j, i] = sugeno_infer(vals[MetalContent.name], vals[Contamination.name],
                                   vals[Repairability.name], vals[Hazard.name])[0]
    plt.figure(figsize=(6.4, 4.8))
    im = plt.imshow(Z, origin="lower", aspect="auto",
                    extent=[x.min(), x.max(), y.min(), y.max()])
    plt.colorbar(im, label="Decision Score (0–100)")
    plt.xlabel(xvar.name); plt.ylabel(yvar.name); plt.title("Decision Surface (2D slice)")
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()

def save_all_figures(output_dir: str = "."):
    plot_variable(MetalContent,  f"{output_dir}/mf_metal.png")
    plot_variable(Contamination, f"{output_dir}/mf_contamination.png")
    plot_variable(Repairability, f"{output_dir}/mf_repairability.png")
    plot_variable(Hazard,        f"{output_dir}/mf_hazard.png")
    plot_slice_contamination(80, 4, 2, f"{output_dir}/slice_contamination.png")
    plot_surface_2d(MetalContent, Contamination,
                    {MetalContent.name: 50, Contamination.name: 50, Repairability.name: 5, Hazard.name: 5},
                    f"{output_dir}/surface_metal_vs_contamination.png")
    plot_surface_2d(Repairability, Hazard,
                    {MetalContent.name: 60, Contamination.name: 40, Repairability.name: 5, Hazard.name: 5},
                    f"{output_dir}/surface_repairability_vs_hazard.png")
