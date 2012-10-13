.. role:: envvar(literal)
.. role:: command(literal)
.. role:: file(literal)
.. role:: ref(title-reference)
.. _OVERVIEW:

Overview
========

This software framework implements a NURBS-based Galerkin finite
element method, geared towards solving the weak form of partial
differential equations. This method is popularly known as isogeometric
analysis (the *IGA* in PetIGA).  We base this framework on PETSc_, the
*Portable, Extensible Toolkit for Scientific Computation*. PETSc is a
collection of algorithms and data structures for the solution of
scientific problems, particularly those modeled by partial
differential equations. PETSc is written to be applicable to a range
of problem sizes, including large-scale simulations where high
performance parallel is a must. PETSc uses the message-passing
interface (MPI) model for communication, but provides high-level
interfaces with collective semantics so that typical users rarely have
to make message-passing calls directly. PetIGA can be thought of as an
extension of PETSc, which adds the NURBS discretization capability and
the integration of forms. The PetIGA framework is intended for
researchers in the numeric solution of PDEs who have applications
which require extensive computational resources.

To install this package, you will need to install PETSc. Our framework
works against both the release and development versions. Once this is
installed make sure that your environment variables
:envvar:`PETSC_DIR` and :envvar:`PETSC_ARCH` are properly
defined. Compilation of PetIGA only requires that you enter the main
directory and type :command:`make`. See :ref:`INSTALL` for more
detailed installation instructions. After installation, we invite you
to explore sample applications located in the :file:`demo/`
directory. A more complete tutorial can be found in :ref:`TUTORIAL`.

The best way to begin coding with this framework is to examine the
demo programs. Our framework can be used to solve linear, nonlinear,
time dependent, or time dependent nonlinear problems. The burden which
the user has is to provide the evaluation of the linear form
(right-hand side, or residual of a nonlinear problem) at a Gauss point
as well as the bilinear form (left-hand side or Jacobian of the
nonlinear residual). This philosphy enables researchers to focus on
the physics of the problem and all-but-ignore issues of parallelism
and performance. We suggest choosing a demo problem which is of the
type of problem you wish to solve, and studying the required
components.

.. _PETSc: http://www.mcs.anl.gov/petsc/

.. Local Variables:
.. mode: rst
.. End:
