import numpy as np
import sympy as sp


def conduction_1d(q=None,
                  k=None,
                  A=None,
                  T1=None,
                  T2=None,
                  L=None):
    """
    Solves the 1D steady-state conduction
     equation for the unknown variable.
    q = k*A*(T1 - T2)/L
    Provide all variables except one,
      and the function will solve for the missing one.
    """
    # Define symbols
    syms = sp.symbols('q k A T1 T2 L')
    q_sym, k_sym, A_sym, T1_sym, T2_sym, L_sym = syms

    # Equation
    eq = sp.Eq(q_sym, k_sym * A_sym * (T1_sym - T2_sym) / L_sym)

    # Build dictionary of knowns
    knowns = {
        q_sym: q,
        k_sym: k,
        A_sym: A,
        T1_sym: T1,
        T2_sym: T2,
        L_sym: L
    }

    # Find the unknown variable
    unknowns = [var for var, val in knowns.items()
                if val is None]

    if len(unknowns) != 1:
        raise ValueError("One variable must be None to solve for it.")

    unknown = unknowns[0]

    # Solve for the unknown
    solution = sp.solve(eq.subs({k: v for k, v in knowns.items()
                                 if v is not None}), unknown)

    return float(solution[0])


def conduction_radial(q=None,
                      k=None,
                      L=None,
                      T1=None,
                      T2=None,
                      r1=None,
                      r2=None):
    """
    Solves the radial steady-state conduction
      equation for the unknown variable.
    q = 2 * π * k * L * (T1 - T2) / ln(r2/r1)
    Provide all variables except one,
      and the function will solve for the missing one.
    """
    # Define symbols
    syms = sp.symbols('q k L T1 T2 r1 r2')
    q_sym, k_sym, L_sym, T1_sym, T2_sym, r1_sym, r2_sym = syms

    # Equation
    eq = sp.Eq(q_sym,
               2 * sp.pi * k_sym * L_sym * (T1_sym - T2_sym
                                            ) / sp.ln(r2_sym / r1_sym))

    # Build dictionary of knowns
    knowns = {
        q_sym: q,
        k_sym: k,
        L_sym: L,
        T1_sym: T1,
        T2_sym: T2,
        r1_sym: r1,
        r2_sym: r2
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


def conduction_spherical(q=None,
                         k=None,
                         L=None,
                         T1=None,
                         T2=None,
                         r1=None,
                         r2=None):
    """
    Solves the spherical steady-state
     conduction equation for the unknown variable.
    q = 4 * π * k * L * (T1 - T2) / (1/r1 - 1/r2)
    Provide all variables except one,
      and the function will solve for the missing one.
    """
    # Define symbols
    syms = sp.symbols('q k L T1 T2 r1 r2')
    q_sym, k_sym, L_sym, T1_sym, T2_sym, r1_sym, r2_sym = syms

    # Equation
    eq = sp.Eq(q_sym,
               4 * sp.pi * k_sym * L_sym * (T1_sym - T2_sym
                                            ) / (1/r1_sym - 1/r2_sym))

    # Build dictionary of knowns
    knowns = {
        q_sym: q,
        k_sym: k,
        L_sym: L,
        T1_sym: T1,
        T2_sym: T2,
        r1_sym: r1,
        r2_sym: r2
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
