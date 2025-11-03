import numpy as np
import sympy as sp

def convection(q=None, h=None, A=None, T_s=None, T_amb=None):
    """
    Solves the convection heat transfer equation for the unknown variable.
    q = h * A * (T_s - T_amb)
    Provide all variables except one, and the function will solve for the missing one.
    """
    # Define symbols
    q_sym, h_sym, A_sym, T_s_sym, T_amb_sym = sp.symbols('q h A T_s T_amb')
    
    # Equation
    eq = sp.Eq(q_sym, h_sym * A_sym * (T_s_sym - T_amb_sym))

    # Build dictionary of knowns
    knowns = {
        q_sym: q,
        h_sym: h,
        A_sym: A,
        T_s_sym: T_s,
        T_amb_sym: T_amb
    }
    
    # Find the unknown variable
    unknowns = [var for var, val in knowns.items() if val is None]
    
    if len(unknowns) != 1:
        raise ValueError("One variable must be None to solve for it.")
    
    unknown = unknowns[0]
    
    # Solve for the unknown
    solution = sp.solve(eq.subs({k: v for k, v in knowns.items() if v is not None}), unknown)
    
    return float(solution[0])

def forced_convection_flat_plate(NuL=None, ReL=None, Pr=None, L=None, k=None):
    
    if NuL is None:
        if ReL > 1e3 and ReL < 5e5:
            NuL = 0.664 * ReL**0.5 * Pr**(1/3)  # Laminar
        elif ReL >= 5e5:
            NuL = 0.036 * ReL**(4/5) * Pr**(1/3)  # Turbulent
    
    h_hat = NuL * k / L
    return h_hat, NuL

def Prandtl(Pr=None, Cp=None, mu=None, k=None, nu=None, alpha=None):
    """
    Calculates the Prandtl number.
    Pr = Cp * mu / k = nu / alpha
    Provide all variables except one, and the function will solve for the missing one.
    """
    # Define symbols
    Pr_sym, Cp_sym, mu_sym, k_sym, nu_sym, alpha_sym = sp.symbols('Pr Cp mu k nu alpha')
    
    # Equation
    eq = sp.Eq(Pr_sym, Cp_sym * mu_sym / k_sym)
    
    # Alternate form
    eq_alpha = sp.Eq(Pr_sym, nu_sym / alpha_sym)

    # Build dictionary of knowns
    knowns = {
        Pr_sym: Pr,
        Cp_sym: Cp,
        mu_sym: mu,
        k_sym: k,
        nu_sym: nu,
        alpha_sym: alpha
    }
    
    # Find the unknown variable
    unknowns = [var for var, val in knowns.items() if val is None]
    
    unknown = unknowns[0]
    
    # Choose appropriate equation
    if unknown in [Pr_sym, Cp_sym, mu_sym, k_sym]:
        eq_to_use = eq
    else:
        eq_to_use = eq_alpha
    
    # Solve for the unknown
    solution = sp.solve(eq_to_use.subs({k: v for k, v in knowns.items() if v is not None}), unknown)
    
    return float(solution[0])

def free_convection_vertical_plate(NuL=None, RaL=None, L=None, k=None, Pr=None):
    """
    Solves for convective heat transfer coefficient
    for free convection over a vertical plate.
    NuL = h_hat * L / k
    h_hat = NuL * k / L
    Returns h_hat and NuL
    """
    
    if NuL is None and RaL < 1e9 and RaL > 1e4:
        NuL = 0.59 * RaL**0.25 #laminar
    elif NuL is None and RaL < 1e13 and RaL >= 1e9:
        NuL = 0.10 * RaL**(1/3) #turbulent
    elif NuL is None and RaL < 1e9:
        NuL = 0.68 + (0.670 * RaL**0.25) / (1 + (0.492 / Pr)**(9/16))**(4/9) #laminar with Pr correction
    elif NuL is None:
        NuL = (0.825 + (0.387 * RaL**(1/6)) / (1 + (0.492 / Pr)**(9/16))**(8/27))**2 #turbulent with Pr correction

    h_hat = NuL * k / L
    return h_hat, NuL

def free_convection_horizontal_plate(NuL=None, RaL=None, L=None, k=None, A_s = None, P=None):
    """
    Solves for convective heat transfer coefficient
    for free convection over a horizontal plate.
    NuL = h_hat * L / k
    h_hat = NuL * k / L
    Returns h_hat and NuL
    """
    if L is None:
        L = A_s / P

    if NuL is None and RaL < 2e7 and RaL > 1e5:
        NuL = 0.54 * RaL**0.25
    elif NuL is None and RaL > 2e7 and RaL < 3e10:
        NuL = 0.14 * RaL**(1/3)
    elif NuL is None and RaL > 3e5 and RaL < 1e10:
        NuL = 0.27 * RaL**0.25
    else:
        raise ValueError("RaL out of range for correlations.")

    h_hat = NuL * k / L
    return h_hat, NuL


def free_convection_horizontal_plate(NuL=None, RaL=None, L=None, k=None, A_s = None, P=None):

    """
    Solves for convective heat transfer coefficient
    for free convection over a horizontal plate.
    NuL = h_hat * L / k
    h_hat = NuL * k / L
    Returns h_hat and NuL
    """
    if L is None:
        L = A_s / P

    if NuL is None and RaL < 2e7 and RaL > 1e5:
        NuL = 0.54 * RaL**0.25
    elif NuL is None and RaL > 2e7 and RaL < 3e10:
        NuL = 0.14 * RaL**(1/3)
    elif NuL is None and RaL > 3e5 and RaL < 1e10:
        NuL = 0.27 * RaL**0.25
    else:
        raise ValueError("RaL out of range for correlations.")

    h_hat = NuL * k / L
    return h_hat, NuL

def free_convection_cylinder(NuD=None, RaD=None, D=None, k=None, Pr=None, nu=None, alpha=None, mu=None, Cp=None):
    """
    Solves for convective heat transfer coefficient
    for free convection over a horizontal cylinder.
    NuD = h_hat * D / k
    h_hat = NuD * k / D
    Returns h_hat and NuD
    """
    if Pr is None and nu is not None and alpha is not None:
        Pr = nu / alpha
    elif Pr is None and mu is not None and Cp is not None and k is not None:
        Pr = Cp * mu / k
    
    if NuD is None and RaD < 1e12:
        NuD = (0.60 + (0.387*RaD**(1/6))/(1+(0.559/Pr)**(9/16))**(4/9))**2 #laminar
    else:
        raise ValueError("RaD out of range for correlations.")

    h_hat = NuD * k / D
    return h_hat, NuD

def free_convection_sphere(NuD=None, RaD=None, D=None, k=None, Pr=None):
    """
    Solves for convective heat transfer coefficient
    for free convection over a sphere.
    NuD = h_hat * D / k
    h_hat = NuD * k / D
    Returns h_hat and NuD
    """
    
    if NuD is None and RaD < 1e11:
        NuD = 2 + (0.589*RaD**(1/4)) / (1 + (0.469/Pr)**(9/16))**(4/9)
    else:
        raise ValueError("RaD out of range for correlations.")

    h_hat = NuD * k / D
    return h_hat, NuD

def rayleigh(T_surf, T_inf, L, nu, alpha, g=9.81):
    """
    Calculates the Rayleigh number for free convection.
    RaL = g * beta * (T_s - T_inf) * L^3 / (nu * alpha)
    """
    beta = 1 / ((T_surf + T_inf) / 2)  # Approximate thermal expansion coefficient for ideal gases
    RaL = g * beta * (T_surf - T_inf) * L**3 / (nu * alpha)
    return RaL