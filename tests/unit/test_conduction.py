import math
import pytest

import conduction


def test_conduction_1d_solve_for_q():
    q = conduction.conduction_1d(
        q=None, k=10.0, A=2.0, T1=110.0, T2=100.0, L=1.0
    )
    assert q == pytest.approx(200.0)


def test_conduction_1d_solve_for_T2():
    T2 = conduction.conduction_1d(
        q=200.0, k=10.0, A=2.0, T1=110.0, T2=None, L=1.0
    )
    assert T2 == pytest.approx(100.0)


def test_conduction_1d_requires_exactly_one_unknown():
    # Two unknowns (q and T2) should raise
    with pytest.raises(ValueError):
        conduction.conduction_1d(
            q=None, k=10.0, A=2.0, T1=110.0, T2=None, L=1.0
        )


def test_conduction_radial_cylinder_solve_for_q():
    q = conduction.conduction_radial_cylinder(
        q=None, k=15.0, L=2.0, T1=200.0, T2=100.0, r1=0.05, r2=0.10
    )
    expected = (
        2.0 * math.pi * 15.0 * 2.0 * (200.0 - 100.0) /
        math.log(0.10 / 0.05)
    )
    assert q == pytest.approx(expected)
