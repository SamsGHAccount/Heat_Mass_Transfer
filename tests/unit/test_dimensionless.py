import math
import pytest

import forced_convection
import free_convection
import transient


def test_reynolds_solve_for_Re_and_v():
    Re = forced_convection.Reynolds(
        Re=None, rho=1.2, v=2.0, L=0.5, mu=1.8e-5
    )
    expected_Re = 1.2 * 2.0 * 0.5 / 1.8e-5
    assert Re == pytest.approx(expected_Re)

    v = forced_convection.Reynolds(
        Re=1e5, rho=1.2, v=None, L=0.5, mu=1.8e-5
    )
    expected_v = 1e5 * 1.8e-5 / (1.2 * 0.5)
    assert v == pytest.approx(expected_v)


def test_fourier_number_solve_Fo_and_t():
    Fo = transient.fourier_number(Fo=None, alpha=1e-5, t=60.0, L=0.1)
    expected_Fo = 1e-5 * 60.0 / 0.1**2
    assert Fo == pytest.approx(expected_Fo)

    t = transient.fourier_number(Fo=Fo, alpha=1e-5, t=None, L=0.1)
    assert t == pytest.approx(60.0)


def test_rayleigh_number():
    Ra = free_convection.rayleigh(
        T_surf=350.0, T_inf=300.0, L=0.5, nu=1.5e-5, alpha=2.0e-5
    )
    beta = 1.0 / ((350.0 + 300.0) / 2.0)
    expected = (
        9.81 * beta * (350.0 - 300.0) * 0.5**3 /
        (1.5e-5 * 2.0e-5)
    )
    assert Ra == pytest.approx(expected)
