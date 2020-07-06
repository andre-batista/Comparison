"""The Born Iterative Method.

This module implements the Born Iterative Method [1] as a derivation of
Solver class. The object contains an object of a forward solver and
one linear inverse solver object. The method solves the nonlinear
inverse problem iteratively.

References
----------
.. [1] Wang, Y. M., and Weng Cho Chew. "An iterative solution of the
   two‐dimensional electromagnetic inverse scattering problem."
   International Journal of Imaging Systems and Technology 1.1 (1989):
   100-108.
"""

import copy as cp
import time as tm

import library_v2.configuration as cfg
import library_v2.inputdata as ipt
import library_v2.results as rst
import library_v2.solver as slv
import library_v2.forward as fwr
import library_v2.inverse as inv


class BornIterativeMethod(slv.Solver):
    """The Born Interative Method (BIM).

    This class implements a classical nonlinear inverse solver [1]. The
    method is based on coupling forward and inverse solvers in an
    iterative process. Therefore, it depends on the definition of a
    forward solver implementation and an linear inverse one.

    Attributes
    ----------
        forward : :class:`forward.Forward`:
            An implementation of the abstract class which defines a
            forward method which solves the total electric field.

        inverse : :class:`inverse.Inverse`:
            An implementation of the abstract class which defines method
            for solving the linear inverse scattering problem.

        MAX_IT : int
            The number of iterations.

    References
    ----------
    .. [1] Wang, Y. M., and Weng Cho Chew. "An iterative solution of the
       two‐dimensional electromagnetic inverse scattering problem."
       International Journal of Imaging Systems and Technology 1.1 (1989):
       100-108.
    """

    name = 'Born Iterative Method'
    forward = fwr.ForwardSolver(cfg.Configuration(name='', frequency=float))
    inverse = inv.Inverse()
    MAX_IT = int()

    def __init__(self, configuration, version, forward_solver, inverse_solver,
                 maximum_iterations=10):
        """Create the object.

        Parameters
        ----------
            configuration : :class:`configuration.Configuration`
                It may be either an object of problem configuration or
                a string to a pre-saved file or a 2-tuple with the file
                name and path, respectively.

            version : str
                A string naming the version of this method. It may be
                useful when using different implementation of forward
                and inverse solvers.

            forward_solver : :class:`forward.Forward`
                An implementation of the abstract class Forward which
                defines a method for computing the total intern field.

            inverse_solver : :class:`inverse.Inverse`
                An implementation of the abstract class Inverse which
                defines a method for solving the linear inverse problem.

            maximum_iterations : int, default: 10
                Maximum number of iterations.
        """
        super().__init__(configuration)
        self.MAX_IT = maximum_iterations
        self.forward = forward_solver
        self.inverse = inverse_solver
        self.version = version

        if self.forward.configuration is None:
            self.forward.configuration = self.configuration

        if self.inverse.configuration is None:
            self.inverse.configuration = self.configuration

    def solve(self, instance, PRINT_INFO=True):
        """Solve a nonlinear inverse problem.

        Parameters
        ----------
            instance : :class:`inputdata.InputData`
                An object which defines a case problem with scattered
                field and some others information.

            PRINT_INFO : bool
                Print or not the iteration information.
        """
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
        self.execution_time = 0.

        for it in range(self.MAX_IT):

            iteration_message = 'Iteration: %d - ' % it
            tic = tm.time()
            self.inverse.solve(solution)
            self.forward.solve(solution)
            self.execution_time = self.execution_time + (tm.time()-tic)
            result.update_error(instance, scattered_field=solution.es,
                                total_field=solution.et,
                                relative_permittivity_map=solution.epsilon_r,
                                conductivity_map=solution.sigma)
            iteration_message = result.last_error_message(instance,
                                                          iteration_message)
            print(iteration_message)

        result.es = solution.es
        result.et = solution.et
        result.epsilon_r = solution.epsilon_r
        result.sigma = solution.sigma
        result.execution_time = self.execution_time
