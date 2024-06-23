from . import advection_2d
import numpy as np
import os

# Load dictionary with expected solutions
expected_sols_dict = np.load('expected_sols.npy',allow_pickle=True).item()

def error(test_name,**kwargs):
    assert test_name in expected_sols_dict.keys(), f"Test name {test_name} not found in expected_sols.npy"
    claw = advection_2d.setup(outdir=None,**kwargs)
    claw.run()
    qtest = claw.frames[claw.num_output_times].state.get_q_global().reshape([-1])
    dx = claw.solution.domain.grid.delta[0]
    dy = claw.solution.domain.grid.delta[1]
    qexpected = expected_sols_dict[test_name]
    diff_L1 = dx*dy*np.sum(np.abs(qtest-qexpected))
    return diff_L1

class TestAdvection2D:
    def test_classic_dimensional_split(self):
        assert abs(error(test_name='classic_dim_split', solver_type='classic',
                         dimensional_split=True))<1e-5

    def test_classic_unsplit_no_trans(self):
        assert abs(error(test_name='classic_unsplit_no_trans', solver_type='classic',
                         dimensional_split=False,transverse_waves=0))<1e-5

    def test_classic_unsplit_trans_inc(self):
        assert abs(error(test_name='classic_unsplit_trans_inc',solver_type='classic',
                         dimensional_split=False,transverse_waves=1))<1e-5
        
    def test_classic_unsplit_trans_cor(self):
        assert abs(error(test_name='classic_unsplit_trans_cor', solver_type='classic',
                         dimensional_split=False,transverse_waves=2))<1e-5
    
    def test_sharpclaw(self):
        assert abs(error(test_name='sharpclaw', solver_type='sharpclaw'))<1e-5
