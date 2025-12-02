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

def forced_convection_sphere(NuD=None, ReD=None, Pr=None, D=None, k=None):
    if NuD is None:
        if ReD < 450:
            NuD = 2 + 0.6 * ReD**0.5 * Pr**(1/3)  # Laminar
        elif ReD >= 450:
            NuD = 2 + (0.4 * ReD**0.5 + 0.06 * ReD**(2/3)) * Pr**0.4 / (1 + (0.4/Pr)**(2/3))**0.25  # Turbulent
    
    h_hat = NuD * k / D
    return h_hat, NuD

def forced_convection_cylinder(NuD=None, ReD=None, Pr=None, D=None, k=None):
    if NuD is None:
        NuD = 0.3 + (0.62 * ReD**0.5 * Pr**(1/3)) / (1 + (0.4/Pr)**(2/3))**0.25 * (1 + (ReD/282000)**(5/8))**(4/5)
    
    h_hat = NuD * k / D
    return h_hat, NuD



import sympy as sp

def Prandtl(Pr=None, Cp=None, mu=None, k=None, nu=None, alpha=None):
    """
    Solve any one variable from the Prandtl-number relations

        1) Pr = Cp * mu / k
        2) Pr = nu / alpha

    Only the variables that actually appear in the chosen relation
    need to be non-None. Exactly one of those variables must be None.
    """
    Pr_s, Cp_s, mu_s, k_s, nu_s, alpha_s = sp.symbols('Pr Cp mu k nu alpha')

    eq1 = sp.Eq(Pr_s, Cp_s * mu_s / k_s)   # uses (Pr, Cp, mu, k)
    eq2 = sp.Eq(Pr_s, nu_s / alpha_s)      # uses (Pr, nu, alpha)

    knowns = {
        Pr_s: Pr,
        Cp_s: Cp,
        mu_s: mu,
        k_s: k,
        nu_s: nu,
        alpha_s: alpha,
    }

    equations = [
        (eq1, (Pr_s, Cp_s, mu_s, k_s)),
        (eq2, (Pr_s, nu_s, alpha_s)),
    ]

    candidates = []
    for eq, syms in equations:
        unknowns = [s for s in syms if knowns[s] is None]
        if len(unknowns) != 1:
            continue
        unknown = unknowns[0]
        # all the other vars in this equation must be known
        if any(knowns[s] is None for s in syms if s is not unknown):
            continue
        candidates.append((eq, unknown))

    if not candidates:
        raise ValueError(
            "Could not determine which Prandtl relation to use. "
            "For Pr = Cp*mu/k, provide 3 of {Pr, Cp, mu, k}. "
            "For Pr = nu/alpha, provide 2 of {Pr, nu, alpha}."
        )

    # Prefer Cp*mu/k form if both work
    eq_use, unknown = candidates[0]
    for eq, unk in candidates:
        if eq is eq1:
            eq_use, unknown = eq, unk
            break

    subs = {sym: val for sym, val in knowns.items() if val is not None}
    sol = sp.solve(eq_use.subs(subs), unknown)
    if not sol:
        raise ValueError("No solution found for Prandtl number.")

    result = float(sol[0].evalf())

    # Tell the GUI which variable we solved for
    symbol_to_name = {
        Pr_s: "Pr",
        Cp_s: "Cp",
        mu_s: "mu",
        k_s: "k",
        nu_s: "nu",
        alpha_s: "alpha",
    }
    Prandtl.last_solved = symbol_to_name.get(unknown)

    return result



def Reynolds(Re=None, rho=None, v=None, L=None, mu=None, nu=None):
    """
    Solve any one variable from the Reynolds-number relations

        1) Re = rho * v * L / mu
        2) Re = v * L / nu

    Only the variables that actually appear in the chosen relation
    need to be non-None. Exactly one of those variables should be None.
    """
    Re_s, rho_s, v_s, L_s, mu_s, nu_s = sp.symbols('Re rho v L mu nu')

    eq1 = sp.Eq(Re_s, rho_s * v_s * L_s / mu_s)  # (Re, rho, v, L, mu)
    eq2 = sp.Eq(Re_s, v_s * L_s / nu_s)          # (Re, v, L, nu)

    knowns = {
        Re_s: Re,
        rho_s: rho,
        v_s: v,
        L_s: L,
        mu_s: mu,
        nu_s: nu,
    }

    equations = [
        (eq1, (Re_s, rho_s, v_s, L_s, mu_s)),
        (eq2, (Re_s, v_s, L_s, nu_s)),
    ]

    candidates = []
    for eq, syms in equations:
        unknowns = [s for s in syms if knowns[s] is None]
        if len(unknowns) != 1:
            continue
        unknown = unknowns[0]
        if any(knowns[s] is None for s in syms if s is not unknown):
            continue
        candidates.append((eq, unknown))

    if not candidates:
        raise ValueError(
            "Could not determine which Reynolds relation to use. "
            "For Re = rho*v*L/mu, provide 4 of {Re, rho, v, L, mu}. "
            "For Re = v*L/nu, provide 3 of {Re, v, L, nu}."
        )

    # Prefer rho*v*L/mu form if it works
    eq_use, unknown = candidates[0]
    for eq, unk in candidates:
        if eq is eq1:
            eq_use, unknown = eq, unk
            break

    subs = {sym: val for sym, val in knowns.items() if val is not None}
    sol = sp.solve(eq_use.subs(subs), unknown)
    if not sol:
        raise ValueError("No solution found for Reynolds number.")

    result = float(sol[0].evalf())

    symbol_to_name = {
        Re_s: "Re",
        rho_s: "rho",
        v_s: "v",
        L_s: "L",
        mu_s: "mu",
        nu_s: "nu",
    }
    Reynolds.last_solved = symbol_to_name.get(unknown)

    return result