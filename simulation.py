from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import sys
import importlib

# Import files from the src directory 

# Sets path to src directory
THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import the files from source
import conduction
import forced_convection
import radiation

GeometryKind = Literal["plane", "cylinder", "sphere"]


@dataclass
class Geometry:
    kind: GeometryKind
    # Conduction resistance for the solid region(s) [K/W]
    R_cond: float
    # Inside / outside areas [m2] used for convection & radiation
    A_in: float
    A_out: float
    # Characteristic lengths for convection on each side [m]
    L_char_in: float
    L_char_out: float


@dataclass
class FluidProps:
    name: str
    rho: float #density [kg/m3]
    mu: float #dynamic viscosity [Pa*s]
    k: float #thermal conductivity [W/(m*K)]
    Cp: float #specific heat [J/(kg*K)]
    v: float #characteristic velocity [m/s]


def prompt_float(msg: str) -> float:
    """
    Helper function that prompts the user for a value
    """
    while True:
        text = input(msg + " ")
        try:
            return float(text)
        except ValueError:
            print("Please enter a numeric value.")


def prompt_int(msg: str, minimum: int = 1) -> int:
    '''
    Helper function that prompts the user for an integer value
    greater than or equal to minimum.

    Docstring for prompt_int

    :param msg: Description
    :type msg: str
    :param minimum: Description
    :type minimum: int
    :return: Description
    :rtype: int
    '''
    while True:
        text = input(msg + " ")
        try:
            value = int(text)
        except ValueError:
            print("Please enter an integer value.")
            continue
        if value < minimum:
            print(f"Please enter a value >= {minimum}.")
            continue
        return value


def prompt_choice(msg: str, options):
    '''
    Helper function that prompts the user to choose one of the possible options.

    Docstring for prompt_choice
    :param msg: Description
    :type msg: str
    :param options: Description
    '''
    opts = {str(o).lower(): o for o in options}
    while True:
        text = input(msg + f" {list(opts.keys())}: ").strip().lower()
        if text in opts:
            return opts[text]
        print(f"Please choose one of {list(opts.keys())}.")

def build_geometry() -> Geometry:
    """
    Asks the user for a specific geometry and how many layers before prompting for 
    properties of each layer and areas needed for convection and radiation.
    """
    kind: GeometryKind = prompt_choice(
        "Choose geometry of system", ["plane", "cylinder", "sphere"]
    )

    if kind == "plane":
        A = prompt_float("Enter wall area normal to heat flow, A [m2]:")
        n_layers = prompt_int("Number of solid layers (1 for single layer):", minimum=1)

        R_cond = 0.0
        for i in range(1, n_layers + 1):
            print(f"\nLayer {i} for plane wall")
            k_i = prompt_float("  Thermal conductivity k [W/(m*K)]:")
            L_i = prompt_float("  Thickness L [m]:")
            R_i = L_i / (k_i * A)
            R_cond += R_i

        # For convection, we need a characteristic length along the flow.
        L_char = prompt_float(
            "\nEnter the length along the flow for convection (plate length), L_char [m]:"
        )

        return Geometry(
            kind="plane",
            R_cond=R_cond,
            A_in=A,
            A_out=A,
            L_char_in=L_char,
            L_char_out=L_char,
        )

    elif kind == "cylinder":
        L = prompt_float("Enter cylinder length, L [m]:")
        r_inner = prompt_float("Enter inner radius, r_inner [m]:")

        n_layers = prompt_int("Number of solid layers (1 for single material):", minimum=1)

        R_cond = 0.0
        r_prev = r_inner
        for i in range(1, n_layers + 1):
            print(f"\nLayer {i} for cylinder")
            k_i = prompt_float("Thermal conductivity k [W/(m*K)]:")
            r_out = prompt_float("Outer radius for this layer [m] (must be > previous radius):")
            if r_out <= r_prev:
                raise ValueError("Outer radius must be greater than inner radius for each layer.")
            R_i = math.log(r_out / r_prev) / (2.0 * math.pi * k_i * L)
            R_cond += R_i
            r_prev = r_out

        r_outer = r_prev
        A_in = 2.0 * math.pi * r_inner * L
        A_out = 2.0 * math.pi * r_outer * L

        D_in = 2.0 * r_inner
        D_out = 2.0 * r_outer

        return Geometry(
            kind="cylinder",
            R_cond=R_cond,
            A_in=A_in,
            A_out=A_out,
            L_char_in=D_in,
            L_char_out=D_out,
        )

    else:  # sphere
        r_inner = prompt_float("Enter inner radius of spherical wall, r_inner [m]:")
        n_layers = prompt_int("Number of solid layers (1 for single material):", minimum=1)

        R_cond = 0.0
        r_prev = r_inner
        for i in range(1, n_layers + 1):
            print(f"\nLayer {i} for sphere")
            k_i = prompt_float("Thermal conductivity k [W/(m*K)]:")
            r_out = prompt_float("Outer radius for this layer [m] (must be > previous radius):")
            if r_out <= r_prev:
                raise ValueError("Outer radius must be greater than inner radius for each layer.")
            R_i = (1.0 / r_prev - 1.0 / r_out) / (4.0 * math.pi * k_i)
            R_cond += R_i
            r_prev = r_out

        r_outer = r_prev
        A_in = 4.0 * math.pi * r_inner ** 2
        A_out = 4.0 * math.pi * r_outer ** 2

        D_in = 2.0 * r_inner
        D_out = 2.0 * r_outer

        return Geometry(
            kind="sphere",
            R_cond=R_cond,
            A_in=A_in,
            A_out=A_out,
            L_char_in=D_in,
            L_char_out=D_out,
        )

def get_fluid_props(label: str) -> FluidProps:
    print(f"\nEnter {label} fluid properties (evaluated at a representative film temperature).")
    rho = prompt_float("  Density rho [kg/m3]:")
    mu = prompt_float("  Dynamic viscosity mu [Pa*s]:")
    k = prompt_float("  Thermal conductivity k [W/(m*K)]:")
    Cp = prompt_float("  Specific heat Cp [J/(kg*K)]:")
    v = prompt_float("  Flow velocity v [m/s]:")
    return FluidProps(name=label, rho=rho, mu=mu, k=k, Cp=Cp, v=v)

def compute_convection(
    geom: Geometry,
    side: Literal["inside", "outside"],
    fluid: FluidProps,
) -> Tuple[float, float, float, str]:
    """
    Compute h, Re, Pr and regime for a given side and geometry using
    correlations available in forced_convection.py.

    Returns (h, Re, Pr, regime).
    """
    if side == "inside":
        L_char = geom.L_char_in
        A = geom.A_in
    else:
        L_char = geom.L_char_out
        A = geom.A_out

    # Non-dimensional numbers
    Re = fluid.rho * fluid.v * L_char / fluid.mu
    Pr = fluid.Cp * fluid.mu / fluid.k

    if geom.kind == "plane":
        # External flow over flat plate
        h, Nu = forced_convection.forced_convection_flat_plate(
            NuL=None, ReL=Re, Pr=Pr, L=L_char, k=fluid.k
        )
        regime = "laminar" if Re < 5.0e5 else "turbulent"
    elif geom.kind == "cylinder":
        D = L_char
        h, Nu = forced_convection.forced_convection_cylinder(
            NuD=None, ReD=Re, Pr=Pr, D=D, k=fluid.k
        )
        # Churchill–Bernstein correlation is widely used across regimes;
        # we use Re around 2e5 as a rough laminar–turbulent divider.
        regime = "laminar" if Re < 2.0e5 else "turbulent"
    else:  # sphere
        D = L_char
        h, Nu = forced_convection.forced_convection_sphere(
            NuD=None, ReD=Re, Pr=Pr, D=D, k=fluid.k
        )
        regime = "laminar" if Re < 450.0 else "turbulent"

    print(
        f"{side.capitalize()} side: Re = {Re:.3e}, Pr = {Pr:.3f}, "
        f"Nusselt = {Nu:.3f}, regime = {regime}, h = {h:.3f} W/(m2*K)"
    )
    return h, Re, Pr, regime

def compute_radiation_coefficient(A_out: float) -> float:
    """
    Ask for emissivity and a reference absolute temperature and compute
    an effective radiation coefficient for a gray body in a large enclosure.
    """
    print("\nRadiation (gray body in a large enclosure).")
    epsilon = prompt_float("  Surface emissivity epsilon (0–1):")
    T_ref = prompt_float(
        "Reference absolute temperature for radiation coefficient, T_ref [K] (mean surface temp):"
    )

    h_r = radiation.radiation_h_gray_enclosure(h_r=None, epsilon=epsilon, T_s=T_ref)
    print(f"Effective radiation coefficient h_r = {h_r:.3f} W/(m2*K)")
    return h_r

def main() -> None:
    print("Heat & Mass Transfer – Whole Steady-State Simulation")
    print("All inputs are SI units. Temperatures for conduction/convection "
          "may be in C or K (only differences matter).")
    print("Radiation uses an absolute temperature you will specify explicitly.\n")

    geom = build_geometry()

    print("\nFluid properties:")
    fluid_in = get_fluid_props("inside")
    fluid_out = get_fluid_props("outside")

    print("\nConvection coefficients from forced convection correlations:")
    h_in, Re_in, Pr_in, reg_in = compute_convection(geom, "inside", fluid_in)
    h_out, Re_out, Pr_out, reg_out = compute_convection(geom, "outside", fluid_out)

    print("\nRadiation coefficient :")
    h_r = compute_radiation_coefficient(geom.A_out)

    # Thermal resistances (K/W)
    R_conv_in = 1.0 / (h_in * geom.A_in)
    R_parallel_out = 1.0 / ((h_out + h_r) * geom.A_out)
    R_total = R_conv_in + geom.R_cond + R_parallel_out

    print("\nSummary of thermal resistances:")
    print(f"  R_conv_inside = {R_conv_in:.6e} K/W")
    print(f"  R_cond_total  = {geom.R_cond:.6e} K/W")
    print(f"  R_out (conv + rad, parallel) = {R_parallel_out:.6e} K/W")
    print(f"  R_total       = {R_total:.6e} K/W\n")

    # Choose what to solve for
    what = prompt_choice(
        "What would you like to solve for? (q = heat rate, T = unknown bulk temperature)",
        ["q", "T"],
    )

    if what == "q":
        print("\nSolving for total heat-transfer rate q.")
        T_hot = prompt_float("Hot-side bulk temperature [C or K]:")
        T_cold = prompt_float("Cold-side bulk temperature [C or K]:")

        q = (T_hot - T_cold) / R_total
        print(f"\nRESULT: q = {q:.3f} W (positive means heat flows from hot to cold side).")

    else:
        which_T = prompt_choice(
            "Which bulk temperature is unknown? (inside or outside)",
            ["inside", "outside"],
        )
        q_given = prompt_float("Enter known magnitude of heat-transfer rate q [W] (positive for heat flowing from inside to outside):")

        if which_T == "inside":
            T_out = prompt_float("Known outside bulk temperature [C or K]:")
            T_in = T_out + q_given * R_total
            print(f"\nRESULT: Inside bulk temperature T_inside = {T_in:.3f} (same units as input).")
        else:
            T_in = prompt_float("Known inside bulk temperature [C or K]:")
            T_out = T_in - q_given * R_total
            print(f"\nRESULT: Outside bulk temperature T_outside = {T_out:.3f} (same units as input).")

    print("\nDone.")

if __name__ == "__main__":
    main()