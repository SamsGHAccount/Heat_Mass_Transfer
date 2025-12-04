import numpy as np
import sympy as sp


def fourier_number(Fo=None, alpha=None, t=None, L=None):
    """
    Fourier number for transient 1D conduction in a plane wall:

    Fo = alpha * t / L**2
    """
    Fo_s, alpha_s, t_s, L_s = sp.symbols('Fo alpha t L')

    eq = sp.Eq(Fo_s, alpha_s * t_s / L_s**2)

    knowns = {
        Fo_s: Fo,
        alpha_s: alpha,
        t_s: t,
        L_s: L,
    }

    unknowns = [sym for sym, val in knowns.items() if val is None]
    if len(unknowns) != 1:
        raise ValueError("Exactly one of Fo, alpha, t, L must be None.")

    unknown = unknowns[0]

    subs = {sym: float(val) for sym, val in knowns.items() if val is not None}
    solutions = sp.solve(eq.subs(subs), unknown)

    if not solutions:
        raise ValueError("No solution found for the requested variable.")

    # Prefer positive solution when multiple are returned (e.g. solving for L)
    sol_value = None
    for s in solutions:
        try:
            s_val = float(sp.N(s))
        except TypeError:
            continue
        if s_val > 0:
            sol_value = s_val
            break
    if sol_value is None:
        sol_value = float(sp.N(solutions[0]))

    return sol_value


def flat_wall_long_time(T=None, x=None, T1=None, T2=None, L=None):
    """
    Long-time solution for a flat wall suddenly heated on one side
    (Fourier number Fo = alpha t / L^2 >> 1).

    Temperature distribution becomes linear between T1 and T2:
    T(x) = T1 - (T1 - T2) * x / L
    """
    T_s, x_s, T1_s, T2_s, L_s = sp.symbols('T x T1 T2 L')

    eq = sp.Eq(T_s, T1_s - (T1_s - T2_s) * x_s / L_s)

    knowns = {
        T_s: T,
        x_s: x,
        T1_s: T1,
        T2_s: T2,
        L_s: L,
    }

    unknowns = [sym for sym, val in knowns.items() if val is None]
    if len(unknowns) != 1:
        raise ValueError("Exactly one of T, x, T1, T2, L must be None.")

    unknown = unknowns[0]
    subs = {sym: float(val) for sym, val in knowns.items() if val is not None}
    solutions = sp.solve(eq.subs(subs), unknown)

    if not solutions:
        raise ValueError("No solution found for the requested variable.")

    return float(sp.N(solutions[0]))


def flat_wall_short_time(T=None, x=None, t=None, T1=None, T2=None, alpha=None):
    """
    Short-time solution for a flat wall suddenly heated on one side
    (Fourier number Fo << 1).
    Error-function solution:
    (T(x, t) - T2) / (T1 - T2) = 1 - erf( x / (2 * sqrt(alpha * t)))
    """
    T_s, x_s, t_s, T1_s, T2_s, alpha_s = sp.symbols('T x t T1 T2 alpha', positive=True)

    phi = x_s / (2 * sp.sqrt(alpha_s * t_s))
    eq = sp.Eq((T_s - T2_s) / (T1_s - T2_s), 1 - sp.erf(phi))

    knowns = {
        T_s: T,
        x_s: x,
        t_s: t,
        T1_s: T1,
        T2_s: T2,
        alpha_s: alpha,
    }

    unknowns = [sym for sym, val in knowns.items() if val is None]
    if len(unknowns) != 1:
        raise ValueError("Exactly one of T, x, t, T1, T2, alpha must be None.")

    unknown = unknowns[0]
    subs = {sym: float(val) for sym, val in knowns.items() if val is not None}

    solutions = sp.solve(eq.subs(subs), unknown)

    if not solutions:
        raise ValueError("No solution found for the requested variable.")

    return float(sp.N(solutions[0]))


def flat_wall_full_series(T=None, x=None, t=None, T1=None, T2=None,
                          alpha=None, L=None, n_terms=50):
    """
    Exact series solution for a flat wall suddenly heated on one side, valid for all times:
    (T - T2) / (T1 - T2) = 1 - x/L - (2/pi) * sum_{n=1...infinity} (1/n) * sin(n*pi*x/L) * exp(-n^2*pi^2*alpha*t/L^2)
    """
    if T is not None:
        raise ValueError("flat_wall_full_series only solves for T; "
                         "leave T blank (None) and supply all other parameters.")

    if any(v is None for v in (x, t, T1, T2, alpha, L)):
        raise ValueError("x, t, T1, T2, alpha, and L must all be provided.")

    # Use plain floats for the numerical series
    x = float(x)
    t = float(t)
    T1 = float(T1)
    T2 = float(T2)
    alpha = float(alpha)
    L = float(L)

    theta = 1.0 - x / L
    Fo = alpha * t / (L**2)

    for n in range(1, int(n_terms) + 1):
        term = (1.0 / n) * np.sin(n * np.pi * x / L) * np.exp(- (n**2) * (np.pi**2) * Fo)
        theta -= (2.0 / np.pi) * term

    T_val = T2 + (T1 - T2) * theta
    return float(T_val)