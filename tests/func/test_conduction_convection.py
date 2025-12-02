import pytest

import forced_convection
import conduction


def test_conduction_equals_convection_in_simple_wall_case():
    A = 1.0
    k = 20.0
    L = 0.1
    T_hot = 400.0
    T_surface = 350.0
    T_amb = 300.0

    # Heat conducted through the wall
    q_cond = conduction.conduction_1d(
        q=None, k=k, A=A, T1=T_hot, T2=T_surface, L=L
    )

    # Choose h such that convection carries away the same heat
    h = q_cond / (A * (T_surface - T_amb))

    q_conv = forced_convection.convection(
        q=None, h=h, A=A, T_s=T_surface, T_amb=T_amb
    )

    # End-to-end, conduction and convection should match
    assert q_conv == pytest.approx(q_cond)
