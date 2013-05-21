PetIGA: A framework for high performance Isogeometric Analysis
==============================================================


Overview
--------

This software framework implements a NURBS-based Galerkin finite
element method (FEM), popularly known as `isogeometric analysis
<http://wikipedia.org/wiki/Isogeometric_analysis>`_ (IGA). It is
heavily based on `PETSc <http://www.mcs.anl.gov/petsc/>`_, the
*Portable, Extensible Toolkit for Scientific Computation*. PETSc is a
collection of algorithms and data structures for the solution of
scientific problems, particularly those modeled by partial
differential equations (PDEs). PETSc is written to be applicable to a
range of problem sizes, including large-scale simulations where high
performance parallel is a must. PetIGA can be thought of as an
extension of PETSc, which adds the NURBS discretization capability and
the integration of forms. The PetIGA framework is intended for
researchers in the numeric solution of PDEs who have applications
which require extensive computational resources.


Installation
------------

After `installing PETSc
<http://www.mcs.anl.gov/petsc/documentation/installation.html>`_,
set appropriate values for ``PETSC_DIR`` and ``PETSC_ARCH`` in your
environment::

  $ export PETSC_DIR=/home/user/petsc-3.4.0
  $ export PETSC_ARCH=arch-linux2-c-debug

Clone the `Mercurial <http://mercurial.selenic.com/>`_ repository
hosted at `Bitbucket <https://bitbucket.org/dalcinl/petiga>`_ ::

  $ hg clone https://bitbucket.org/dalcinl/PetIGA

Finally, enter PetIGA top level directory and use ``make`` to compile
the code and build the PetIGA library::

  $ cd PetIGA
  $ make all
  $ make test


Acknowledgments
---------------

This project was partially supported by the Center for Numerical
Porous Media, Division of Computer, Electrical, and Mathematical
Sciences & Engineering (`CEMSE <http://cemse.kaust.edu.sa/>`_), King
Abdullah University of Science and Technology (`KAUST
<http://www.kaust.edu.sa/>`_).
