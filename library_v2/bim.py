"""Type the title of the module here.

Give a brief explanation of the module.
"""

import copy as cp

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

    def __init__(self, configuration, version, forward_solver, inverse_solver,
                 maximum_iterations=10):
        """Summarize the method."""
        super().__init__(configuration)
        self.MAX_IT = maximum_iterations
        self.forward = forward_solver
        self.inverse = inverse_solver
        self.version = version

    def solve(self, instance, PRINT_INFO=True):
        """Summarize method."""
        result = rst.Results(instance.name + '_' + self.version,
                             method_name=self.name,
                             configuration_filename=self.configuration.name,
                             configuration_filepath=self.configuration.path,
                             inputdata_filename=instance.name)

        if PRINT_INFO:
            self._print_title(instance)
            print('Iterations: %d' % self.MAX_IT)
            print('Forward solver: ' + self.forward.name)
            print('Inverse solver: ' + self.inverse.name)
            self.inverse.print_parametrization()

        solution = cp.deepcopy(instance)
        solution.et = self.forward.incident_field(instance.resolution)

        for it in range(self.MAX_IT):

            self.inverse.solve(solution)
            self.forward.solve(solution)
            result.update_error(instance, scattered_field=solution.es,
                                total_field=solution.et,
                                relative_permittivity_map=solution.epsilon_r,
                                conductivity_map=solution.sigma)

        result.es = solution.es
        result.et = solution.et
        result.epsilon_r = solution.epsilon_r
        result.sigma = solution.sigma
