import pytest

import radiation


def test_stefan_boltzmann_solve_E_for_black_surface():
    E = radiation.stefan_boltzmann(E=None, T=300.0, epsilon=1.0)
    sigma = 5.67e-8
    expected = sigma * 300.0**4
    assert E == pytest.approx(expected)


def test_total_heat_transfer_coefficient_adds_components():
    h_total = radiation.total_heat_transfer_coefficient(
        h_total=None, h_r=2.5, h_c=7.5
    )
    assert h_total == pytest.approx(10.0)

    # Solve for one component as well
    h_r = radiation.total_heat_transfer_coefficient(
        h_total=10.0, h_r=None, h_c=7.5
    )
    assert h_r == pytest.approx(2.5)


def test_radiation_h_gray_enclosure_forward():
    h_r = radiation.radiation_h_gray_enclosure(
        h_r=None, epsilon=0.9, T_s=400.0
    )
    sigma = 5.67e-8
    expected = 4.0 * 0.9 * sigma * 400.0**3
    assert h_r == pytest.approx(expected)
