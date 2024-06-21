from . import advection_1d_nonunif
import numpy as np
from clawpack.pyclaw.util import check_diff
import os

def error(**kwargs):
    """
    Compute difference between initial and final solutions.
    This should vanish due to the periodic boundary conditions.
    The L1 norm for the error accounts for the nonuniform grid.
    """
    claw = advection_1d_nonunif.setup(outdir=None,**kwargs)
    claw.run()

    # tests are done across the entire domain of q normally
    q0 = claw.frames[0].state.get_q_global()
    qfinal = claw.frames[claw.num_output_times].state.get_q_global()
    q0 = q0.reshape([-1])
    qfinal = qfinal.reshape([-1])

    physical_nodes = claw.frames[0].state.grid.p_nodes
    dx=claw.solution.domain.grid.delta[0]
    comp_nodes = claw.frames[0].state.grid.c_nodes
    jac_mapc2p = np.diff(physical_nodes)/np.diff(comp_nodes)
    return np.sum(abs(dx*jac_mapc2p*(qfinal-q0)))

class TestAdvection1D:
    def test_python_classic(self):
        assert abs(error(kernel_language='Python',solver_type='classic')-0.04305825207098881)<1e-6

    def test_fortran_classic(self):
        assert abs(error(kernel_language='Fortran',solver_type='classic')-0.04305825207098882)<1e-6
