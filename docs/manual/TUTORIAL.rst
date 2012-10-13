.. role:: option(literal)
.. role:: file(literal)
.. _TUTORIAL:

Tutorial
========

**Objective:** At the end of this tutorial you will understand how to
run, query, and change different components of the Laplace solver.

**Assumptions:** I assume that you have both PETSc and PetIGA
configured and compiled.

Compilation and Execution
-------------------------

We will use the Laplace code included in the :file:`$PETIGA_DIR/demo/`
directory to explain and highlight features of using PetIGA to solve
partial differential equations. First enter this directory and build
this application by typing::

    make Laplace

This code has been written to solve the Laplace equation on the unit
domain in 1,2, or 3 spatial dimensions. The boundary conditions are
Dirichlet on the left and free Neumann on the right, chosen such that
the exact solution is *u = 1* no matter what spatial dimension
is selected. To run this code, simply type::

    ./Laplace -print_error

This will solve the Laplace problem using default discretization
choices and print the *L2* error of the solution.

Query Components
----------------

The natural question is then how does one inquire what the code
did. What discretization was used? Which solver? All of this can be
determined without opening the source file. PETSc was built on the
philosophy that solver components should be changeable from the
command line, without requiring the code to be recompiled. This means
that there are many options available. To see some of them type::

    ./Laplace -help

A long list of options will print along with short descriptions of
what they control. Perusing this list is educational as it gives you
an idea of just how much can be controlled from the command line. To
see information about the default discretization, type::

    ./Laplace -iga_view

which will produce::

    IGA: dimension=3  dofs/node=1  geometry=no  rational=no
    Axis 0: periodic=0  degree=2  quadrature=3  processors=1  nodes=18  elements=16
    Axis 1: periodic=0  degree=2  quadrature=3  processors=1  nodes=18  elements=16
    Axis 2: periodic=0  degree=2  quadrature=3  processors=1  nodes=18  elements=16
    Partitioning - nodes:    sum=5832  min=5832  max=5832  max/min=1
    Partitioning - elements: sum=4096  min=4096  max=4096  max/min=1

This view command tells you information about the function space that
was chosen. We used a 3D quadratic polynomial space with no mapped
geometries. If run in parallel, :option:`-iga_view` will also provide
information about how the problem was divided. For example, on 8
processors::

    IGA: dimension=3  dofs/node=1  geometry=no  rational=no
    Axis 0: periodic=0  degree=2  quadrature=3  processors=2  nodes=18  elements=16
    Axis 1: periodic=0  degree=2  quadrature=3  processors=2  nodes=18  elements=16
    Axis 2: periodic=0  degree=2  quadrature=3  processors=2  nodes=18  elements=16
    Partitioning - nodes:    sum=5832  min=729  max=729  max/min=1
    Partitioning - elements: sum=4096  min=512  max=512  max/min=1

The axis lines reveal that each axis was divided onto 2
processors. The partitioning information is included to provide
details of how balanced the partitioning was in terms of nodes and
elements. In this case, both are perfectly balanced.

We can also learn about which solver was utilized by running the code
with the following command::

     ./Lapace -ksp_view

which generates::

    KSP Object: 1 MPI processes
      type: gmres
        GMRES: restart=30, using Classical (unmodified) Gram-Schmidt Orthogonalization with no iterative refinement
        GMRES: happy breakdown tolerance 1e-30
      maximum iterations=10000, initial guess is zero
      tolerances:  relative=1e-05, absolute=1e-50, divergence=10000
      left preconditioning
      using PRECONDITIONED norm type for convergence test
    PC Object: 1 MPI processes
      type: ilu
        ILU: out-of-place factorization
        0 levels of fill
        tolerance for zero pivot 2.22045e-14
        using diagonal shift to prevent zero pivot
        matrix ordering: natural
        factor fill ratio given 1, needed 1
          Factored matrix follows:
            Matrix Object:         1 MPI processes
              type: seqaij
              rows=5832, cols=5832
              package used to perform factorization: petsc
              total: nonzeros=592704, allocated nonzeros=592704
              total number of mallocs used during MatSetValues calls =0
                not using I-node routines
      linear system matrix = precond matrix:
      Matrix Object:   1 MPI processes
        type: seqaij
        rows=5832, cols=5832
        total: nonzeros=592704, allocated nonzeros=592704
        total number of mallocs used during MatSetValues calls =0
          not using I-node routines

The acronym KSP represents the PETSc abstraction of Krylov subspace
methods. In this case, the method used was GMRES with ILU(0) as
preconditioner. We see that convergence is being determined by a
relative tolerance reduction of 5 orders of magnitude or 10,000
iterations.

We can also monitor the progress of the iteration solver by running
with the :option:`-ksp_monitor` option. This yields::

       0 KSP Residual norm 4.608063258286e+01
       1 KSP Residual norm 1.035767637586e+01
       2 KSP Residual norm 5.117674236377e+00
       3 KSP Residual norm 3.310266416640e+00
       4 KSP Residual norm 1.885825259760e+00
       5 KSP Residual norm 5.137381534630e-01
       6 KSP Residual norm 1.255873067131e-01
       7 KSP Residual norm 3.447800238703e-02
       8 KSP Residual norm 1.023523179223e-02
       9 KSP Residual norm 1.904532606704e-03
      10 KSP Residual norm 4.066469650297e-04

or more concisely we could just use :option:`-ksp_converged_reason`::

    Linear solve converged due to CONVERGED_RTOL iterations 10

Changing Components
-------------------

Different components of the Laplace solver may be changed from the
commandline. For example, if you run the Laplace code with
:option:`-help` again, locate a block of options for the Laplace
problem::

    Laplace Options -------------------------------------------------
      -dim <3>: dimension (Laplace.c)
      -N <16>: number of elements (Laplace.c)
      -p <2>: polynomial order (Laplace.c)
      -C <-1>: global continuity order (Laplace.c)

The numbers in brackets are the default values. The default
discretization is a 3D quadratic space consisting of 16x16x16
elements. The global continuity order defaulting to -1 does not
reflect that the spaces are discontinuous. It is an internal flag to
set the continuity to *p-1*. Our framework then internally builds the
knot vectors which correspond to this space. If we run::

    ./Laplace -dim 2 -N 64 -p 4 -C 0 -iga_view

we get::

    IGA: dimension=2  dofs/node=1  geometry=no  rational=no
    Axis 0: periodic=0  degree=4  quadrature=5  processors=1  nodes=257  elements=64
    Axis 1: periodic=0  degree=4  quadrature=5  processors=1  nodes=257  elements=64
    Partitioning - nodes:    sum=66049  min=66049  max=66049  max/min=1
    Partitioning - elements: sum=4096  min=4096  max=4096  max/min=1

which corresponds to a 64x64 mesh of *C^0* quartics. Similarly the
solver components can be changed from the command line. For example,
we can solve the system using CG and Jacobi by::

    ./Laplace -ksp_type cg -pc_type jacobi -ksp_view

which produces::

    KSP Object: 1 MPI processes
      type: cg
      maximum iterations=10000, initial guess is zero
      tolerances:  relative=1e-05, absolute=1e-50, divergence=10000
      left preconditioning
      using PRECONDITIONED norm type for convergence test
    PC Object: 1 MPI processes
      type: jacobi
      linear system matrix = precond matrix:
      Matrix Object:   1 MPI processes
        type: seqaij
        rows=5832, cols=5832
        total: nonzeros=592704, allocated nonzeros=592704
        total number of mallocs used during MatSetValues calls =0
          not using I-node routines

This tutorial highlights a feature of using PetIGA to solve PDEs--you
immediately have access to a wide variety of expert solvers and
preconditioners. Furthermore, you have query tools to examine and
study your problems for when they fail.

.. Local Variables:
.. mode: rst
.. End:
