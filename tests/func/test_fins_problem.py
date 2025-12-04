import math
import pytest

import fins


def test_infinite_rectangular_fin_end_to_end():
    # Geometry and properties
    W = 0.02      # m
    d = 0.001     # m
    L = 0.05      # m (not directly used here, but included for context)
    k = 200.0     # W/m-K
    h = 25.0      # W/m^2-K
    T0 = 373.0    # K
    T_inf = 293.0 # K

    # Use the helper functions in the fins module
    A = fins.fin_cross_section(A=None, W=W, d=d)
    P = fins.fin_perimeter(P=None, W=W, d=d, use_approx=False)
    m = fins.fin_m(m=None, h=h, P=P, k=k, A=A, d=None, use_rect_approx=False)
    q = fins.fin_heat_transfer_infinite(
        q=None, k=k, A=A, m=m, T0=T0, T_inf=T_inf
    )

    # Verify against the textbook formulas in a single expression
    A_expected = W * d
    P_expected = 2.0 * (W + d)
    m_expected = math.sqrt(h * P_expected / (k * A_expected))
    q_expected = k * A_expected * m_expected * (T0 - T_inf)

    assert A == pytest.approx(A_expected)
    assert P == pytest.approx(P_expected)
    assert m == pytest.approx(m_expected)
    assert q == pytest.approx(q_expected)
