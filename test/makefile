ALL: test-all

CFLAGS          = #-g3 -Wall -Wextra -Wno-unused-parameter #-Wconversion
FFLAGS          = #-g3 -Wall -Wextra -fcheck=all
CPPFLAGS        =
FPPFLAGS        = 
LOCDIR          = test/
EXAMPLESC       = Test_SNES_2D.c
EXAMPLESF       = 
MANSEC          = IGA

topdir := $(shell cd .. && pwd)
PETIGA_DIR ?= $(topdir)
include ${PETIGA_DIR}/conf/petigavariables
include ${PETIGA_DIR}/conf/petigarules

Test_SNES_2D: Test_SNES_2D.o chkopts
	${CLINKER} -o $@ $< ${PETIGA_LIB}
	${RM} -f $<

OPTS=-nox -malloc_debug -malloc_dump
runex1a_1:
	-@${MPIEXEC} -n 1 ./Test_SNES_2D ${OPTS} -N 8 -p 1 -iga_mat_type aij
	-@${MPIEXEC} -n 1 ./Test_SNES_2D ${OPTS} -N 8 -p 1 -iga_mat_type baij
	-@${MPIEXEC} -n 1 ./Test_SNES_2D ${OPTS} -N 8 -p 1 -iga_mat_type sbaij -pc_type none
	-@${MPIEXEC} -n 1 ./Test_SNES_2D ${OPTS} -N 4 -p 1 -iga_mat_type dense
runex1a_4:
	-@${MPIEXEC} -n 4 ./Test_SNES_2D ${OPTS} -N 8 -p 1 -iga_mat_type aij
	-@${MPIEXEC} -n 4 ./Test_SNES_2D ${OPTS} -N 8 -p 1 -iga_mat_type baij
	-@${MPIEXEC} -n 4 ./Test_SNES_2D ${OPTS} -N 8 -p 1 -iga_mat_type sbaij -pc_type none
	-@${MPIEXEC} -n 4 ./Test_SNES_2D ${OPTS} -N 4 -p 1 -iga_mat_type dense
runex1b_1:
	-@${MPIEXEC} -n 1 ./Test_SNES_2D ${OPTS} -p 2
runex1b_4:
	-@${MPIEXEC} -n 4 ./Test_SNES_2D ${OPTS} -p 2
runex1c_1:
	-@${MPIEXEC} -n 1 ./Test_SNES_2D ${OPTS} -p 3
runex1c_4:
	-@${MPIEXEC} -n 4 ./Test_SNES_2D ${OPTS} -p 3

Test_SNES_2D = Test_SNES_2D.PETSc  \
	       runex1a_1 runex1a_4 \
	       runex1b_1 runex1b_4 \
	       runex1c_1 runex1c_4 \
	       Test_SNES_2D.rm

TESTEXAMPLES_C = $(Test_SNES_2D)
TESTEXAMPLES_FORTRAN =
test-all:
	-@${OMAKE} tree ACTION=testexamples_C PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR}
	-@if [ "${FC}" != "" ]; then \
          ${OMAKE} tree ACTION=testexamples_Fortran PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR}; \
          fi
.PHONY: test-all

test-build: Test_SNES_2D.PETSc
	-@if [ "${PETSC_WITH_BATCH}" != "" ]; then \
	    echo "Running with batch filesystem; to test run test/Test_SNES_2D with" ; \
	    echo "your systems batch system"; \
	  elif [ "${MPIEXEC}" = "/bin/false" ]; then \
	    echo "*mpiexec not found*. Please run test/Test_SNES_2D manually"; \
	  elif [ -f Test_SNES_2D ]; then \
	    ${MPIEXEC} -n 1 ./Test_SNES_2D -nox -malloc_debug -malloc_dump; \
	    if [ "${MPIEXEC}" != "${PETSC_DIR}/bin/mpiexec.uni" ]; then \
	      ${MPIEXEC} -n 2 ./Test_SNES_2D -nox -malloc_debug -malloc_dump; \
	      ${MPIEXEC} -n 3 ./Test_SNES_2D -nox -malloc_debug -malloc_dump; \
	      ${MPIEXEC} -n 4 ./Test_SNES_2D -nox -malloc_debug -malloc_dump; \
	    fi; \
	    ${OMAKE} Test_SNES_2D.rm PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR}; \
	   fi
.PHONY:test-build

include ${PETIGA_DIR}/conf/petigatest
