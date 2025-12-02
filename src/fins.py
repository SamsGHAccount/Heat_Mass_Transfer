import sympy as sp
import numpy as np  # kept for consistency with other modules


def _solve_one_from_equations(
    equations,
    knowns,
    prefer_eq=None,
    error_message="Could not determine which relation to use.",
):
    """
    Utility to solve a single unknown from one of several algebraic relations.

    Parameters
    ----------
    equations : list of (Eq, tuple(symbols))
        Each entry is a SymPy equation together with the symbols that appear in it.
    knowns : dict
        Mapping SymPy symbol -> numeric value or None.
    prefer_eq : Eq or None
        If multiple relations could be used, this one is preferred when present.
    error_message : str
        Message for ValueError when no usable relation is found.

    Returns
    -------
    result : float
    unknown : SymPy symbol that was solved for.
    """
    candidates = []

    for eq, syms in equations:
        # Which variables in THIS equation are unknown?
        unknowns = [s for s in syms if knowns.get(s) is None]
        if len(unknowns) != 1:
            continue
        unknown = unknowns[0]
        # All other variables in this equation must be known
        if any(knowns.get(s) is None for s in syms if s is not unknown):
            continue
        candidates.append((eq, unknown))

    if not candidates:
        raise ValueError(error_message)

    # Prefer the specified equation if possible
    eq_use, unknown = candidates[0]
    if prefer_eq is not None:
        for eq, unk in candidates:
            if eq is prefer_eq:
                eq_use, unknown = eq, unk
                break

    subs = {sym: val for sym, val in knowns.items() if val is not None}
    sol = sp.solve(eq_use.subs(subs), unknown)
    if not sol:
        raise ValueError("No solution found.")
    result = float(sol[0])
    return result, unknown


# ---------------------------------------------------------------------------
# Basic fin geometry
# ---------------------------------------------------------------------------

def fin_cross_section(A=None, W=None, d=None):
    """
    Cross-sectional area of a rectangular fin

        A = W * d

    Provide any two of (A, W, d) and leave the third as None to solve for it.
    """
    A_s, W_s, d_s = sp.symbols("A W d")
    eq = sp.Eq(A_s, W_s * d_s)

    knowns = {A_s: A, W_s: W, d_s: d}
    equations = [(eq, (A_s, W_s, d_s))]

    result, unknown = _solve_one_from_equations(
        equations,
        knowns,
        error_message="Provide exactly two of A, W, d to solve the third.",
    )

    symbol_to_name = {A_s: "A", W_s: "W", d_s: "d"}
    fin_cross_section.last_solved = symbol_to_name.get(unknown)

    return result


def fin_perimeter(P=None, W=None, d=None, use_approx=False):
    """
    Perimeter of a rectangular fin

        Exact:   P = 2 W + 2 d
        Approx:  P ≈ 2 W       (for d << W)

    Parameters
    ----------
    P, W, d : float or None
        Leave exactly one of these as None in the chosen relation.
    use_approx : bool, optional
        If True, use the approximate relation P ≈ 2 W (ignoring d).
        If False (default), use the exact expression.

    Returns
    -------
    The solved variable as a float.
    """
    P_s, W_s, d_s = sp.symbols("P W d")

    eq_exact = sp.Eq(P_s, 2 * W_s + 2 * d_s)
    eq_approx = sp.Eq(P_s, 2 * W_s)

    knowns = {P_s: P, W_s: W, d_s: d}

    equations = []
    # Exact expression
    equations.append((eq_exact, (P_s, W_s, d_s)))
    # Approximate relation ignores d
    equations.append((eq_approx, (P_s, W_s)))

    prefer = eq_approx if use_approx else eq_exact

    result, unknown = _solve_one_from_equations(
        equations,
        knowns,
        prefer_eq=prefer,
        error_message=(
            "Provide all but one of the required variables for either "
            "P = 2W + 2d (exact) or P ≈ 2W (approximate)."
        ),
    )

    symbol_to_name = {P_s: "P", W_s: "W", d_s: "d"}
    fin_perimeter.last_solved = symbol_to_name.get(unknown)

    return result


# ---------------------------------------------------------------------------
# Fin parameter m and heat transfer rate
# ---------------------------------------------------------------------------

def fin_m(m=None, h=None, P=None, k=None, A=None, d=None, use_rect_approx=False):
    """
    Fin parameter m for a straight fin of uniform cross-section.

        m = sqrt(h P / (k A))                (general)
          ≈ sqrt(2 h / (k d))                (thin rectangular fin, d << W)

    Parameters
    ----------
    m, h, P, k, A, d : float or None
        For m = sqrt(hP/(kA)), provide 4 of {m, h, P, k, A}.
        For m ≈ sqrt(2h/(k d)), provide 3 of {m, h, k, d}.
    use_rect_approx : bool, optional
        If True, prefer the thin-rectangular approximation when possible.

    Returns
    -------
    The solved variable as a float.
    """
    m_s, h_s, P_s, k_s, A_s, d_s = sp.symbols("m h P k A d")

    eq_general = sp.Eq(m_s, sp.sqrt(h_s * P_s / (k_s * A_s)))
    eq_rect = sp.Eq(m_s, sp.sqrt(2 * h_s / (k_s * d_s)))

    knowns = {
        m_s: m,
        h_s: h,
        P_s: P,
        k_s: k,
        A_s: A,
        d_s: d,
    }

    equations = [
        (eq_general, (m_s, h_s, P_s, k_s, A_s)),
        (eq_rect, (m_s, h_s, k_s, d_s)),
    ]

    prefer = eq_rect if use_rect_approx else eq_general

    result, unknown = _solve_one_from_equations(
        equations,
        knowns,
        prefer_eq=prefer,
        error_message=(
            "Could not determine which relation to use for m. "
            "For m = sqrt(hP/(kA)), provide 4 of {m, h, P, k, A}. "
            "For m ≈ sqrt(2h/(k d)), provide 3 of {m, h, k, d}."
        ),
    )

    symbol_to_name = {
        m_s: "m",
        h_s: "h",
        P_s: "P",
        k_s: "k",
        A_s: "A",
        d_s: "d",
    }
    fin_m.last_solved = symbol_to_name.get(unknown)

    return result


def fin_heat_transfer_insulated_tip(
    q=None,
    k=None,
    A=None,
    m=None,
    T0=None,
    T_inf=None,
    L=None,
):
    """
    Heat-transfer rate from a straight fin with an insulated tip (dT/dx = 0 at x = L):

        q = k A m (T0 - T_inf) * tanh(m L)

    Provide all variables except one to solve for the missing one.
    """
    q_s, k_s, A_s, m_s, T0_s, Tinf_s, L_s = sp.symbols("q k A m T0 T_inf L")

    eq = sp.Eq(q_s, k_s * A_s * m_s * (T0_s - Tinf_s) * sp.tanh(m_s * L_s))

    knowns = {
        q_s: q,
        k_s: k,
        A_s: A,
        m_s: m,
        T0_s: T0,
        Tinf_s: T_inf,
        L_s: L,
    }

    result, unknown = _solve_one_from_equations(
        [(eq, (q_s, k_s, A_s, m_s, T0_s, Tinf_s, L_s))],
        knowns,
        error_message="Exactly one of q, k, A, m, T0, T_inf, L must be left as None.",
    )

    symbol_to_name = {
        q_s: "q",
        k_s: "k",
        A_s: "A",
        m_s: "m",
        T0_s: "T0",
        Tinf_s: "T_inf",
        L_s: "L",
    }
    fin_heat_transfer_insulated_tip.last_solved = symbol_to_name.get(unknown)

    return result


def fin_heat_transfer_infinite(q=None, k=None, A=None, m=None, T0=None, T_inf=None):
    """
    Heat-transfer rate from a very long fin (m L >> 1):

        q_inf = k A m (T0 - T_inf)

    Provide all variables except one to solve for the missing one.
    """
    q_s, k_s, A_s, m_s, T0_s, Tinf_s = sp.symbols("q k A m T0 T_inf")

    eq = sp.Eq(q_s, k_s * A_s * m_s * (T0_s - Tinf_s))

    knowns = {
        q_s: q,
        k_s: k,
        A_s: A,
        m_s: m,
        T0_s: T0,
        Tinf_s: T_inf,
    }

    result, unknown = _solve_one_from_equations(
        [(eq, (q_s, k_s, A_s, m_s, T0_s, Tinf_s))],
        knowns,
        error_message="Exactly one of q, k, A, m, T0, T_inf must be left as None.",
    )

    symbol_to_name = {
        q_s: "q",
        k_s: "k",
        A_s: "A",
        m_s: "m",
        T0_s: "T0",
        Tinf_s: "T_inf",
    }
    fin_heat_transfer_infinite.last_solved = symbol_to_name.get(unknown)

    return result


def fin_heat_transfer_long_rectangular(
    q=None, k=None, h=None, d=None, W=None, T0=None, T_inf=None
):
    """
    Approximate heat-transfer rate from a very long, thin rectangular fin:

        q_inf ≈ (2 k h d)^{1/2} * W * (T0 - T_inf)

    Provide all variables except one to solve for the missing one.
    """
    q_s, k_s, h_s, d_s, W_s, T0_s, Tinf_s = sp.symbols("q k h d W T0 T_inf")

    eq = sp.Eq(q_s, sp.sqrt(2 * k_s * h_s * d_s) * W_s * (T0_s - Tinf_s))

    knowns = {
        q_s: q,
        k_s: k,
        h_s: h,
        d_s: d,
        W_s: W,
        T0_s: T0,
        Tinf_s: T_inf,
    }

    result, unknown = _solve_one_from_equations(
        [(eq, (q_s, k_s, h_s, d_s, W_s, T0_s, Tinf_s))],
        knowns,
        error_message=(
            "Exactly one of q, k, h, d, W, T0, T_inf must be left as None "
            "for the long rectangular fin approximation."
        ),
    )

    symbol_to_name = {
        q_s: "q",
        k_s: "k",
        h_s: "h",
        d_s: "d",
        W_s: "W",
        T0_s: "T0",
        Tinf_s: "T_inf",
    }
    fin_heat_transfer_long_rectangular.last_solved = symbol_to_name.get(unknown)

    return result


# ---------------------------------------------------------------------------
# Fin efficiency and overall surface performance
# ---------------------------------------------------------------------------

def fin_surface_area(A_f=None, L=None, W=None, d=None, use_approx=False):
    """
    Surface area of a straight rectangular fin:

        A_f = 2 L (W + d)          (general)
            ≈ 2 L W                (for d << W)

    Provide inputs so that exactly one of A_f, L, W, d is left as None in the
    relation that you wish to use.
    """
    A_f_s, L_s, W_s, d_s = sp.symbols("A_f L W d")

    eq_general = sp.Eq(A_f_s, 2 * L_s * (W_s + d_s))
    eq_approx = sp.Eq(A_f_s, 2 * L_s * W_s)

    knowns = {
        A_f_s: A_f,
        L_s: L,
        W_s: W,
        d_s: d,
    }

    equations = [
        (eq_general, (A_f_s, L_s, W_s, d_s)),
        (eq_approx, (A_f_s, L_s, W_s)),
    ]

    prefer = eq_approx if use_approx else eq_general

    result, unknown = _solve_one_from_equations(
        equations,
        knowns,
        prefer_eq=prefer,
        error_message=(
            "Could not determine which relation to use for fin surface area. "
            "For A_f = 2L(W + d) provide 3 of {A_f, L, W, d}. "
            "For A_f ≈ 2LW provide 2 of {A_f, L, W}."
        ),
    )

    symbol_to_name = {
        A_f_s: "A_f",
        L_s: "L",
        W_s: "W",
        d_s: "d",
    }
    fin_surface_area.last_solved = symbol_to_name.get(unknown)

    return result


def fin_efficiency(eta_f=None, q=None, h=None, A_f=None, T0=None, T_inf=None):
    """
    Fin efficiency

        eta_f = q / q_max
        q_max = h A_f (T0 - T_inf)

    Provide known values so that exactly one of (eta_f, q, h, A_f, T0, T_inf)
    is left as None.
    """
    eta_s, q_s, h_s, A_f_s, T0_s, Tinf_s = sp.symbols("eta_f q h A_f T0 T_inf")

    q_max = h_s * A_f_s * (T0_s - Tinf_s)
    eq_eta = sp.Eq(eta_s, q_s / q_max)

    knowns = {
        eta_s: eta_f,
        q_s: q,
        h_s: h,
        A_f_s: A_f,
        T0_s: T0,
        Tinf_s: T_inf,
    }

    result, unknown = _solve_one_from_equations(
        [(eq_eta, (eta_s, q_s, h_s, A_f_s, T0_s, Tinf_s))],
        knowns,
        error_message=(
            "Exactly one of eta_f, q, h, A_f, T0, T_inf must be left as None."
        ),
    )

    symbol_to_name = {
        eta_s: "eta_f",
        q_s: "q",
        h_s: "h",
        A_f_s: "A_f",
        T0_s: "T0",
        Tinf_s: "T_inf",
    }
    fin_efficiency.last_solved = symbol_to_name.get(unknown)

    return result


def fin_total_finned_area(A_fins=None, n=None, A_f=None):
    """
    Total finned surface area for n identical fins:

        A_fins = n * A_f
    """
    A_fins_s, n_s, A_f_s = sp.symbols("A_fins n A_f")

    eq = sp.Eq(A_fins_s, n_s * A_f_s)

    knowns = {
        A_fins_s: A_fins,
        n_s: n,
        A_f_s: A_f,
    }

    result, unknown = _solve_one_from_equations(
        [(eq, (A_fins_s, n_s, A_f_s))],
        knowns,
        error_message="Provide exactly two of A_fins, n, A_f to solve the third.",
    )

    symbol_to_name = {
        A_fins_s: "A_fins",
        n_s: "n",
        A_f_s: "A_f",
    }
    fin_total_finned_area.last_solved = symbol_to_name.get(unknown)

    return result


def fin_base_area_without_fins(A0=None, A_base=None, n=None, A=None):
    """
    Net base area that is not covered by fins:

        A0 = A_base - n * A

    where A is the cross-sectional area of a single fin.

    Provide all but one of (A0, A_base, n, A).
    """
    A0_s, A_base_s, n_s, A_s = sp.symbols("A0 A_base n A")

    eq = sp.Eq(A0_s, A_base_s - n_s * A_s)

    knowns = {
        A0_s: A0,
        A_base_s: A_base,
        n_s: n,
        A_s: A,
    }

    result, unknown = _solve_one_from_equations(
        [(eq, (A0_s, A_base_s, n_s, A_s))],
        knowns,
        error_message="Provide all but one of A0, A_base, n, A to solve for the missing one.",
    )

    symbol_to_name = {
        A0_s: "A0",
        A_base_s: "A_base",
        n_s: "n",
        A_s: "A",
    }
    fin_base_area_without_fins.last_solved = symbol_to_name.get(unknown)

    return result


def fin_total_heat_transfer(
    q_total=None,
    A0=None,
    h0=None,
    eta_f=None,
    A_fins=None,
    h=None,
    T0=None,
    T_inf=None,
):
    """
    Total heat-transfer rate from base + fins:

        q_total = A0 h0 (T0 - T_inf) + eta_f A_fins h (T0 - T_inf)

    Note: the convection coefficients h0 (base) and h (fins) may differ.

    Provide known values so that exactly one of
    (q_total, A0, h0, eta_f, A_fins, h, T0, T_inf) is left as None.
    """
    qtot_s, A0_s, h0_s, eta_s, A_fins_s, h_s, T0_s, Tinf_s = sp.symbols(
        "q_total A0 h0 eta_f A_fins h T0 T_inf"
    )

    eq = sp.Eq(
        qtot_s,
        A0_s * h0_s * (T0_s - Tinf_s) + eta_s * A_fins_s * h_s * (T0_s - Tinf_s),
    )

    knowns = {
        qtot_s: q_total,
        A0_s: A0,
        h0_s: h0,
        eta_s: eta_f,
        A_fins_s: A_fins,
        h_s: h,
        T0_s: T0,
        Tinf_s: T_inf,
    }

    result, unknown = _solve_one_from_equations(
        [(eq, (qtot_s, A0_s, h0_s, eta_s, A_fins_s, h_s, T0_s, Tinf_s))],
        knowns,
        error_message=(
            "Exactly one of q_total, A0, h0, eta_f, A_fins, h, T0, T_inf "
            "must be left as None."
        ),
    )

    symbol_to_name = {
        qtot_s: "q_total",
        A0_s: "A0",
        h0_s: "h0",
        eta_s: "eta_f",
        A_fins_s: "A_fins",
        h_s: "h",
        T0_s: "T0",
        Tinf_s: "T_inf",
    }
    fin_total_heat_transfer.last_solved = symbol_to_name.get(unknown)

    return result