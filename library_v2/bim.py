r"""The Born Iterative Method.

This module implements the Born Iterative Method [1]_ as a derivation of
Solver class. The object contains an object of a forward solver and
one of linear inverse solver. The method solves the nonlinear
inverse problem iteratively. The implemented in
:class:`BornIterativeMethod`

References
----------
.. [1] Wang, Y. M., and Weng Cho Chew. "An iterative solution of the
   two‐dimensional electromagnetic inverse scattering problem."
   International Journal of Imaging Systems and Technology 1.1 (1989):
   100-108.
"""

# Standard libraries
import copy as cp
import time as tm
import numpy as np
import sys

# Developed libraries
import configuration as cfg
import inputdata as ipt
import results as rst
import solver as slv
import forward as fwr
import inverse as inv


class BornIterativeMethod(slv.Solver):
    r"""The Born Interative Method (BIM).

    This class implements a classical nonlinear inverse solver [1]_. The
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
        self.alias = version

        if self.forward.configuration is None:
            self.forward.configuration = self.configuration

        if self.inverse.configuration is None:
            self.inverse.configuration = self.configuration

    def solve(self, instance, print_info=True, print_file=sys.stdout):
        """Solve a nonlinear inverse problem.

        Parameters
        ----------
            instance : :class:`inputdata.InputData`
                An object which defines a case problem with scattered
                field and some others information.

            print_info : bool
                Print or not the iteration information.
        """
        super().solve(instance, print_info, print_file)

        if self.forward.configuration.name != self.configuration:
            self.forward.configuration = cp.deepcopy(self.configuration)
        if self.inverse.configuration.name != self.configuration:
            self.inverse.configuration = cp.deepcopy(self.configuration)

        result = rst.Results(instance.name + '_' + self.alias,
                             method_name=self.alias,
                             configuration_filename=self.configuration.name,
                             configuration_filepath=self.configuration.path,
                             input_filename=instance.name,
                             input_filepath=instance.path)

        if print_info:
            print('Iterations: %d' % self.MAX_IT, file=print_file)
            print('----------------------------------------', file=print_file)
            print(self.forward, file=print_file)
            print('----------------------------------------', file=print_file)
            print(self.inverse, file=print_file)
            print('----------------------------------------', file=print_file)

        # The solution variable will be an object of InputData.
        solution = cp.deepcopy(instance)

        # First-Order Born Approximation
        solution.et = self.forward.incident_field(instance.resolution)

        # If the same object is used for different resolution instances,
        # then some parameters may need to be updated within the inverse
        # solver. So, the next line ensures it:
        self.inverse.reset_parameters()
        self.execution_time = 0.

        for it in range(self.MAX_IT):

            iteration_message = 'Iteration: %d - ' % (it+1)
            tic = tm.time()
            self.inverse.solve(solution)
            self.forward.solve(solution, SAVE_INTERN_FIELD=True)

            # The variable `execution_time` will record only the time
            # expended by the forward and linear routines.
            self.execution_time = self.execution_time + (tm.time()-tic)

            result.update_error(instance, scattered_field=solution.es,
                                total_field=solution.et,
                                relative_permittivity_map=solution.epsilon_r,
                                conductivity_map=solution.sigma)

            iteration_message = result.last_error_message(instance,
                                                          iteration_message)
            if print_info:
                print(iteration_message, file=print_file)

            # This is only emergencial feature for ensuring that the
            # scattered field data received by the inverse solver in the
            # next iteration will not be the one estimated in the last
            # call of the forward solver.
            if it != self.MAX_IT-1:
                solution.es = np.copy(instance.es)

        # Remember: results stores the estimated scattered field. Not
        # the given one.
        result.es = solution.es
        result.et = solution.et
        result.epsilon_r = solution.epsilon_r
        result.sigma = solution.sigma
        result.execution_time = self.execution_time

        return result
