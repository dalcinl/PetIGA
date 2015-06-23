.. role:: envvar(literal)
.. role:: command(literal)
.. role:: file(literal)
.. _INSTALL:

Install
=======

This finite element framework is based on PETSc_. You will need to
download and install this library to use PetIGA.

PETSc
-----

PetIGA works with the release of PETSc as well as PETSc-Dev. For the
most recent release of PETSc, download the compressed tar file from
`this <http://www.mcs.anl.gov/petsc/download/>`_ page. You may also
use the development version of PETSc which can be checked-out using
`Git <http://git-scm.com/>`_ by the following command::

    git clone https://bitbucket.org/petsc/petsc.git

In either case, once you have the library source code downloaded,
enter top-level source directory and run the :file:`configure`
script::

    cd petsc
    ./configure

Once the configure is complete, the output from the script will guide
you in how to finish compiling the library. At this point, note the
values that PETSc assigns to 2 environment variables:
:envvar:`PETSC_DIR` and :envvar:`PETSC_ARCH`. We suggest exporting
these variables in your :file:`.bashrc` or :file:`.profile` file such
that they are always set in your environment. PetIGA will depend on
these variables being set properly. For troubleshooting the
configuring and compilation of PETSc, please refer to its `page
<http://www.mcs.anl.gov/petsc/documentation/installation.html>`_ on
the topic.

PetIGA
------

The development version of PetIGA can be checked-out using `Git
<http://git-scm.com/>`_ by the following command::

    git clone https://bitbucket.org/dalcinl/PetIGA.git

To compile PetIGA, simply enter into the top level directory and type
:command:`make`. PetIGA will compile a library based on the values of
:envvar:`PETSC_DIR` and :envvar:`PETSC_ARCH`. That is, if
:envvar:`PETSC_ARCH` points to a version of PETSc with debug flags,
then PetIGA will be compiled with debug flags. To ensure that the
installation is correct, you can then type :command:`make
test`. PetIGA, like PETSc, uses an environment variable to point to
the location of the library, :envvar:`PETIGA_DIR`. We recommend adding
this variable to your :file:`.bashrc` or :file:`.profile` file as
well.

.. _PETSc: http://www.mcs.anl.gov/petsc/

.. Local Variables:
.. mode: rst
.. End:
