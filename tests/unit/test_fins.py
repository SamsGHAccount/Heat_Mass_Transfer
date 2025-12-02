import math
import pytest

import fins


def test_fin_cross_section_solve_A():
    A = fins.fin_cross_section(A=None, W=0.02, d=0.001)
    assert A == pytest.approx(0.02 * 0.001)


def test_fin_perimeter_exact_and_approx():
    W = 0.02
    d = 0.001

    P_exact = fins.fin_perimeter(P=None, W=W, d=d, use_approx=False)
    assert P_exact == pytest.approx(2 * (W + d))

    P_approx = fins.fin_perimeter(P=None, W=W, d=d, use_approx=True)
    assert P_approx == pytest.approx(2 * W)


def test_fin_m_general_and_rectangular_approx():
    h = 10.0
    P = 0.5
    k = 200.0
    A = 0.01

    m_general = fins.fin_m(
        m=None, h=h, P=P, k=k, A=A, d=None, use_rect_approx=False
    )
    assert m_general == pytest.approx(math.sqrt(h * P / (k * A)))

    d = 0.01
    m_rect = fins.fin_m(
        m=None, h=h, P=None, k=k, A=None, d=d, use_rect_approx=True
    )
    assert m_rect == pytest.approx(math.sqrt(2.0 * h / (k * d)))


def test_fin_surface_area_general_and_approx():
    L = 0.1
    W = 0.02
    d = 0.005

    A_general = fins.fin_surface_area(
        A_f=None, L=L, W=W, d=d, use_approx=False
    )
    assert A_general == pytest.approx(2.0 * L * (W + d))

    A_approx = fins.fin_surface_area(
        A_f=None, L=L, W=W, use_approx=True
    )
    assert A_approx == pytest.approx(2.0 * L * W)
