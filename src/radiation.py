import numpy as np
import sympy as sp

# Stefan–Boltzmann constant in SI units [W/m^2-K^4]
SIGMA_DEFAULT = 5.670e-8


def stefan_boltzmann(E=None, T=None, epsilon=1.0, sigma=SIGMA_DEFAULT):
    """
    Stefan–Boltzmann radiation law for a gray surface:

        E = epsilon * sigma * T^4

    epsilon = 1 corresponds to a black surface.

    Provide any three of (E, T, epsilon, sigma) and leave the remaining
    one as None.
    """
    E_s, T_s, eps_s, sig_s = sp.symbols('E T epsilon sigma')
    eq = sp.Eq(E_s, eps_s * sig_s * T_s**4)

    knowns = {
        E_s: E,
        T_s: T,
        eps_s: epsilon,
        sig_s: sigma,
    }

    unknowns = [sym for sym, val in knowns.items() if val is None]
    if len(unknowns) != 1:
        raise ValueError("Exactly one of E, T, epsilon, sigma must be None.")

    unknown = unknowns[0]
    subs = {sym: float(val) for sym, val in knowns.items() if val is not None}

    solutions = sp.solve(eq.subs(subs), unknown)
    if not solutions:
        raise ValueError("No solution found for the requested variable.")

    return float(sp.N(solutions[0]))


def parallel_black_plates(q=None, A=None, T1=None, T2=None, sigma=SIGMA_DEFAULT):
    """
    Two adjacent, parallel black plates:

        q_12 / A = sigma * (T1^4 - T2^4)
        q = A * sigma * (T1^4 - T2^4)

    Provide any four of (q, A, T1, T2, sigma) and leave the remaining one
    as None.
    """
    q_s, A_s, T1_s, T2_s, sig_s = sp.symbols('q A T1 T2 sigma')
    eq = sp.Eq(q_s, A_s * sig_s * (T1_s**4 - T2_s**4))

    knowns = {
        q_s: q,
        A_s: A,
        T1_s: T1,
        T2_s: T2,
        sig_s: sigma,
    }

    unknowns = [sym for sym, val in knowns.items() if val is None]
    if len(unknowns) != 1:
        raise ValueError("Exactly one of q, A, T1, T2, sigma must be None.")

    unknown = unknowns[0]
    subs = {sym: float(val) for sym, val in knowns.items() if val is not None}
    solutions = sp.solve(eq.subs(subs), unknown)
    if not solutions:
        raise ValueError("No solution found for the requested variable.")
    return float(sp.N(solutions[0]))


def parallel_gray_plates(q=None, A=None, T1=None, T2=None,
                         eps1=None, eps2=None, sigma=SIGMA_DEFAULT):
    """
    Two adjacent, parallel gray plates:

        q_12 / A =
            sigma * (T1^4 - T2^4) / (1/eps1 + 1/eps2 - 1)

    Provide any six of (q, A, T1, T2, eps1, eps2, sigma) and leave the
    remaining one as None.
    """
    q_s, A_s, T1_s, T2_s, e1_s, e2_s, sig_s = sp.symbols('q A T1 T2 eps1 eps2 sigma')
    denom = 1 / e1_s + 1 / e2_s - 1
    eq = sp.Eq(q_s, A_s * sig_s * (T1_s**4 - T2_s**4) / denom)

    knowns = {
        q_s: q,
        A_s: A,
        T1_s: T1,
        T2_s: T2,
        e1_s: eps1,
        e2_s: eps2,
        sig_s: sigma,
    }

    unknowns = [sym for sym, val in knowns.items() if val is None]
    if len(unknowns) != 1:
        raise ValueError("Exactly one of q, A, T1, T2, eps1, eps2, sigma must be None.")

    unknown = unknowns[0]
    subs = {sym: float(val) for sym, val in knowns.items() if val is not None}
    solutions = sp.solve(eq.subs(subs), unknown)
    if not solutions:
        raise ValueError("No solution found for the requested variable.")
    return float(sp.N(solutions[0]))


def small_gray_body_in_large_enclosure(q=None, A1=None, eps1=None,
                                       T1=None, T2=None, sigma=SIGMA_DEFAULT):
    """
    Small gray body in a large enclosure:

        q_12 = A1 * eps1 * sigma * (T1^4 - T2^4)

    Provide any five of (q, A1, eps1, T1, T2, sigma) and leave the
    remaining one as None.
    """
    q_s, A1_s, e1_s, T1_s, T2_s, sig_s = sp.symbols('q A1 eps1 T1 T2 sigma')
    eq = sp.Eq(q_s, A1_s * e1_s * sig_s * (T1_s**4 - T2_s**4))

    knowns = {
        q_s: q,
        A1_s: A1,
        e1_s: eps1,
        T1_s: T1,
        T2_s: T2,
        sig_s: sigma,
    }

    unknowns = [sym for sym, val in knowns.items() if val is None]
    if len(unknowns) != 1:
        raise ValueError("Exactly one of q, A1, eps1, T1, T2, sigma must be None.")

    unknown = unknowns[0]
    subs = {sym: float(val) for sym, val in knowns.items() if val is not None}
    solutions = sp.solve(eq.subs(subs), unknown)
    if not solutions:
        raise ValueError("No solution found for the requested variable.")
    return float(sp.N(solutions[0]))


def black_surfaces_general(q=None, A1=None, F12=None,
                           T1=None, T2=None, sigma=SIGMA_DEFAULT):
    """
    General case for two black surfaces:

        q_12 = A1 * F12 * sigma * (T1^4 - T2^4)

    Provide any five of (q, A1, F12, T1, T2, sigma) and leave the remaining
    one as None.
    """
    q_s, A1_s, F12_s, T1_s, T2_s, sig_s = sp.symbols('q A1 F12 T1 T2 sigma')
    eq = sp.Eq(q_s, A1_s * F12_s * sig_s * (T1_s**4 - T2_s**4))

    knowns = {
        q_s: q,
        A1_s: A1,
        F12_s: F12,
        T1_s: T1,
        T2_s: T2,
        sig_s: sigma,
    }

    unknowns = [sym for sym, val in knowns.items() if val is None]
    if len(unknowns) != 1:
        raise ValueError("Exactly one of q, A1, F12, T1, T2, sigma must be None.")

    unknown = unknowns[0]
    subs = {sym: float(val) for sym, val in knowns.items() if val is not None}
    solutions = sp.solve(eq.subs(subs), unknown)
    if not solutions:
        raise ValueError("No solution found for the requested variable.")
    return float(sp.N(solutions[0]))


def composite_view_factor(F12_star=None, F12=None,
                          eps1=None, eps2=None, A1=None, A2=None):
    """
    Composite view factor for two gray surfaces:

        q_12 = A1 * F12_star * sigma * (T1^4 - T2^4)

    with

        F12_star = 1 / ( 1/F12 + (1/eps1 - 1) + (A1/A2) * (1/eps2 - 1) )

    Provide any five of (F12_star, F12, eps1, eps2, A1, A2) and leave the
    remaining one as None.
    """
    Fstar_s, F12_s, e1_s, e2_s, A1_s, A2_s = sp.symbols('F12_star F12 eps1 eps2 A1 A2')
    denom = 1 / F12_s + (1 / e1_s - 1) + (A1_s / A2_s) * (1 / e2_s - 1)
    eq = sp.Eq(Fstar_s, 1 / denom)

    knowns = {
        Fstar_s: F12_star,
        F12_s: F12,
        e1_s: eps1,
        e2_s: eps2,
        A1_s: A1,
        A2_s: A2,
    }

    unknowns = [sym for sym, val in knowns.items() if val is None]
    if len(unknowns) != 1:
        raise ValueError("Exactly one of F12_star, F12, eps1, eps2, A1, A2 must be None.")

    unknown = unknowns[0]
    subs = {sym: float(val) for sym, val in knowns.items() if val is not None}
    solutions = sp.solve(eq.subs(subs), unknown)
    if not solutions:
        raise ValueError("No solution found for the requested variable.")
    return float(sp.N(solutions[0]))


def gray_surfaces_general(q=None, A1=None, F12_star=None,
                          T1=None, T2=None, sigma=SIGMA_DEFAULT):
    """
    General case for two gray surfaces, using the composite view factor:

        q_12 = A1 * F12_star * sigma * (T1^4 - T2^4)

    Provide any five of (q, A1, F12_star, T1, T2, sigma) and leave the
    remaining one as None.
    """
    q_s, A1_s, Fstar_s, T1_s, T2_s, sig_s = sp.symbols('q A1 F12_star T1 T2 sigma')
    eq = sp.Eq(q_s, A1_s * Fstar_s * sig_s * (T1_s**4 - T2_s**4))

    knowns = {
        q_s: q,
        A1_s: A1,
        Fstar_s: F12_star,
        T1_s: T1,
        T2_s: T2,
        sig_s: sigma,
    }

    unknowns = [sym for sym, val in knowns.items() if val is None]
    if len(unknowns) != 1:
        raise ValueError("Exactly one of q, A1, F12_star, T1, T2, sigma must be None.")

    unknown = unknowns[0]
    subs = {sym: float(val) for sym, val in knowns.items() if val is not None}
    solutions = sp.solve(eq.subs(subs), unknown)
    if not solutions:
        raise ValueError("No solution found for the requested variable.")
    return float(sp.N(solutions[0]))


def radiation_h_coefficient(h_r=None, F12=None,
                            T1=None, T2=None, sigma=SIGMA_DEFAULT):
    """
    Radiation heat-transfer coefficient between two surfaces:

        q_12 = A1 * h_r * (T1 - T2)

        h_r = F12 * sigma * (T1^4 - T2^4) / (T1 - T2)
            = F12 * sigma * (T1^2 + T2^2) * (T1 + T2)

    Provide any four of (h_r, F12, T1, T2, sigma) and leave the remaining
    one as None.
    """
    h_s, F12_s, T1_s, T2_s, sig_s = sp.symbols('h_r F12 T1 T2 sigma')
    eq = sp.Eq(h_s, F12_s * sig_s * (T1_s**4 - T2_s**4) / (T1_s - T2_s))

    knowns = {
        h_s: h_r,
        F12_s: F12,
        T1_s: T1,
        T2_s: T2,
        sig_s: sigma,
    }

    unknowns = [sym for sym, val in knowns.items() if val is None]
    if len(unknowns) != 1:
        raise ValueError("Exactly one of h_r, F12, T1, T2, sigma must be None.")

    unknown = unknowns[0]
    subs = {sym: float(val) for sym, val in knowns.items() if val is not None}
    solutions = sp.solve(eq.subs(subs), unknown)
    if not solutions:
        raise ValueError("No solution found for the requested variable.")
    return float(sp.N(solutions[0]))


def radiation_h_gray_enclosure(h_r=None, epsilon=None, T_s=None,
                               sigma=SIGMA_DEFAULT):
    """
    Approximate radiation heat-transfer coefficient for a gray body
    in a large enclosure:

        h_r ≈ 4 * epsilon * sigma * T_s^3

    Provide any three of (h_r, epsilon, T_s, sigma) and leave the
    remaining one as None.
    """
    h_s, eps_s, T_s_sym, sig_s = sp.symbols('h_r epsilon T_s sigma')
    eq = sp.Eq(h_s, 4 * eps_s * sig_s * T_s_sym**3)

    knowns = {
        h_s: h_r,
        eps_s: epsilon,
        T_s_sym: T_s,
        sig_s: sigma,
    }

    unknowns = [sym for sym, val in knowns.items() if val is None]
    if len(unknowns) != 1:
        raise ValueError("Exactly one of h_r, epsilon, T_s, sigma must be None.")

    unknown = unknowns[0]
    subs = {sym: float(val) for sym, val in knowns.items() if val is not None}
    solutions = sp.solve(eq.subs(subs), unknown)
    if not solutions:
        raise ValueError("No solution found for the requested variable.")
    return float(sp.N(solutions[0]))


def total_heat_transfer_coefficient(h_total=None, h_r=None, h_c=None):
    """
    Total heat-transfer coefficient for a hot surface in a cooler room
    of air (radiation + convection):

        h_total = h_r + h_c

    Provide any two of (h_total, h_r, h_c) and leave the remaining one
    as None.
    """
    htot_s, h_r_s, h_c_s = sp.symbols('h_total h_r h_c')
    eq = sp.Eq(htot_s, h_r_s + h_c_s)

    knowns = {
        htot_s: h_total,
        h_r_s: h_r,
        h_c_s: h_c,
    }

    unknowns = [sym for sym, val in knowns.items() if val is None]
    if len(unknowns) != 1:
        raise ValueError("Exactly one of h_total, h_r, h_c must be None.")

    unknown = unknowns[0]
    subs = {sym: float(val) for sym, val in knowns.items() if val is not None}
    solutions = sp.solve(eq.subs(subs), unknown)
    if not solutions:
        raise ValueError("No solution found for the requested variable.")
    return float(sp.N(solutions[0]))


def parallel_gray_with_shield(q=None, A=None, T1=None, T2=None,
                              eps1=None, eps2=None, eps_s=None,
                              sigma=SIGMA_DEFAULT):
    """
    Radiation heat transfer between two adjacent, parallel gray plates
    with a single radiation shield between them.

    Without shield:

        q_12 / A = sigma * (T1^4 - T2^4) / (1/eps1 + 1/eps2 - 1)

    With a shield of emissivity eps_s:

        q_12 / A = sigma * (T1^4 - T2^4)
                   / (1/eps1 + 1/eps2 - 1 + 2 * (1/eps_s - 1))

    Provide any seven of (q, A, T1, T2, eps1, eps2, eps_s, sigma) and
    leave the remaining one as None.
    """
    q_s, A_s, T1_s, T2_s, e1_s, e2_s, es_s, sig_s = sp.symbols(
        'q A T1 T2 eps1 eps2 eps_s sigma'
    )

    denom = 1 / e1_s + 1 / e2_s - 1 + 2 * (1 / es_s - 1)
    eq = sp.Eq(q_s, A_s * sig_s * (T1_s**4 - T2_s**4) / denom)

    knowns = {
        q_s: q,
        A_s: A,
        T1_s: T1,
        T2_s: T2,
        e1_s: eps1,
        e2_s: eps2,
        es_s: eps_s,
        sig_s: sigma,
    }

    unknowns = [sym for sym, val in knowns.items() if val is None]
    if len(unknowns) != 1:
        raise ValueError(
            "Exactly one of q, A, T1, T2, eps1, eps2, eps_s, sigma must be None."
        )

    unknown = unknowns[0]
    subs = {sym: float(val) for sym, val in knowns.items() if val is not None}
    solutions = sp.solve(eq.subs(subs), unknown)
    if not solutions:
        raise ValueError("No solution found for the requested variable.")
    return float(sp.N(solutions[0]))
