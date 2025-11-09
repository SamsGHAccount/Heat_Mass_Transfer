import numpy as np
import sympy as sp


def convection(q=None,
               h=None,
               A=None,
               T_s=None,
               T_amb=None):
    """
    Solves the convection heat transfer
      equation for the unknown variable.
    q = h * A * (T_s - T_amb)
    Provide all variables except one,
      and the function will solve for the missing one.
    """
    # Define symbols
    syms = sp.symbols('q h A T_s T_amb')
    q_sym, h_sym, A_sym, T_s_sym, T_amb_sym = syms

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
    solution = sp.solve(eq.subs({k: v for k, v in knowns.items()
                                 if v is not None}), unknown)

    return float(solution[0])


def forced_convection_flat_plate(NuL=None,
                                 ReL=None,
                                 Pr=None,
                                 L=None,
                                 k=None):

    if NuL is None:
        if ReL > 1e3 and ReL < 5e5:
            NuL = 0.664 * ReL**0.5 * Pr**(1/3)  # Laminar
        elif ReL >= 5e5:
            NuL = 0.036 * ReL**(4/5) * Pr**(1/3)  # Turbulent

    h_hat = NuL * k / L
    return h_hat, NuL


def forced_convection_sphere(NuD=None,
                             ReD=None,
                             Pr=None,
                             D=None,
                             k=None):
    if NuD is None:
        if ReD < 450:
            NuD = 2 + 0.6 * ReD**0.5 * Pr**(1/3)  # Laminar
        elif ReD >= 450:
            NuD = 2 + (0.4 * ReD**0.5
                       + 0.06 * ReD**(2/3)
                       ) * Pr**0.4 / (1
                                      + (0.4/Pr)**(2/3)
                                      )**0.25  # Turbulent

    h_hat = NuD * k / D
    return h_hat, NuD


def forced_convection_cylinder(NuD=None,
                               ReD=None,
                               Pr=None,
                               D=None,
                               k=None):
    if NuD is None:
        NuD = 0.3 + (0.62 * ReD**0.5 * Pr**(1/3)
                     ) / (1
                          + (0.4/Pr)**(2/3)
                          )**0.25 * (1
                                     + (ReD/282000)**(5/8)
                                     )**(4/5)

    h_hat = NuD * k / D
    return h_hat, NuD


def Prandtl(Pr=None, Cp=None, mu=None, k=None, nu=None, alpha=None):
    """
    Calculates the Prandtl number.
    Pr = Cp * mu / k = nu / alpha
    Provide all variables except one,
      and the function will solve for the missing one.
    """
    # Define symbols
    syms = sp.symbols('Pr Cp mu k nu alpha')
    Pr_sym, Cp_sym, mu_sym, k_sym, nu_sym, alpha_sym = syms

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
    solution = sp.solve(eq_to_use.subs({k: v for k, v in knowns.items()
                                        if v is not None}), unknown)

    return float(solution[0])


def Reynolds(Re=None, rho=None, v=None, L=None, mu=None, nu=None):
    """
    Calculates the Reynolds number.
    Re = rho * v * L / mu = v * L / nu
    Provide all variables except one,
      and the function will solve for the missing one.
    """
    # Define symbols
    syms = sp.symbols('Re rho v L mu nu')
    Re_sym, rho_sym, v_sym, L_sym, mu_sym, nu_sym = syms

    # Equation
    eq = sp.Eq(Re_sym, rho_sym * v_sym * L_sym / mu_sym)

    # Alternate form
    eq_nu = sp.Eq(Re_sym, v_sym * L_sym / nu_sym)

    # Build dictionary of knowns
    knowns = {
        Re_sym: Re,
        rho_sym: rho,
        v_sym: v,
        L_sym: L,
        mu_sym: mu,
        nu_sym: nu
    }

    # Find the unknown variable
    unknowns = [var for var, val in knowns.items() if val is None]

    unknown = unknowns[0]

    # Choose appropriate equation
    if unknown in [Re_sym, rho_sym, v_sym, L_sym, mu_sym]:
        eq_to_use = eq
    else:
        eq_to_use = eq_nu

    # Solve for the unknown
    solution = sp.solve(eq_to_use.subs({k: v for k, v in knowns.items()
                                        if v is not None}), unknown)

    return float(solution[0])
