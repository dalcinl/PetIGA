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


Scripting Support
--------------

PetIGA is designed to be efficient and as such, we do not directly do
things like output VTK files suitable for viewing the solution. We do
have routines which output the discretization information and solution
vectors, but these are in a binary format to minimize I/O time. We
have written a python package, `igakit
<https://bitbucket.org/dalcinl/igakit>`_ which handles post-processing
for visualization as well as geometry generation.


Citation
------

If you find PetIGA helpful in conducting research projects, we would
appreciate a citation to the following article:

  @Article{PetIGA,
    author = 	 {N. Collier, L. Dalcin, V.M. Calo},
    title = 	 {{PetIGA}: High-Performance Isogeometric Analysis},
    journal = 	 {arxiv},
    year = 	 {2013},
    number = 	 {1305.4452},
    note = 	 {http://arxiv.org/abs/1305.4452}
  }


Acknowledgments
---------------

This project was partially supported by the Center for Numerical
Porous Media, Division of Computer, Electrical, and Mathematical
Sciences & Engineering (`CEMSE <http://cemse.kaust.edu.sa/>`_), King
Abdullah University of Science and Technology (`KAUST
<http://www.kaust.edu.sa/>`_).
