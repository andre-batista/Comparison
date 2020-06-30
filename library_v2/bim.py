"""Type the title of the module here.

Give a brief explanation of the module.
"""

import library_v2.configuration as cfg
import library_v2.inputdata as ipt
import library_v2.results as rst
import library_v2.solver as slv
import library_v2.forward as fwr
import library_v2.inverse as inv


class BornIterativeMethod(slv.Solver):
    """Summarize the class."""

    name = 'Born Iterative Method'
    forward = fwr.ForwardSolver(cfg.Configuration(name='', frequency=float))
    inverse = inv.Inverse()
    MAX_IT = int()

    def __init__(self, configuration, forward_solver, inverse_solver,
                 maximum_iterations=10):
        """Summarize the method."""
        super().__init__(configuration)
        self.MAX_IT = maximum_iterations
        self.forward = forward_solver
        self.inverse = inverse_solver

    def solve(self, inputdata, PRINT_INFO=True):
        """Summarize method."""
        if PRINT_INFO:
            self._print_title(inputdata)
            print('Iterations: %d' % self.MAX_IT)
            print('Forward solver: ' + self.forward.name)
            print('Inverse solver: ' + self.inverse.name)
            self.inverse.print_parametrization()

        et = self.forward.incident_field(inputdata.resolution)
