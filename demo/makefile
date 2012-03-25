TARGETS = InputOutput \
	  L2Proj1D L2Proj2D \
          Poisson1D Poisson2D Poisson3D \
          Laplace \
          Poisson \
          PatternFormation \
          CahnHilliard2D \
          NavierStokesVMS
ALL: ${TARGETS}
clean::
	-@${RM} ${TARGETS}

CFLAGS    = #-g3 -Wall -Wextra -Wno-unused-parameter #-Wconversion
FFLAGS    = #-g3 -Wall -Wextra -fcheck=all
CPPFLAGS  =
FPPFLAGS  =
LOCDIR    = demo/
EXAMPLESC = CahnHilliard2D.c
EXAMPLESF =
MANSEC    = IGA

topdir := $(shell cd .. && pwd)
PETIGA_DIR ?= $(topdir)
include ${PETIGA_DIR}/conf/petigavariables
include ${PETIGA_DIR}/conf/petigarules

InputOutput: InputOutput.o chkopts
	${CLINKER} -o $@ $< ${PETIGA_LIB}
	${RM} -f $<
L2Proj1D: L2Proj1D.o chkopts
	${CLINKER} -o $@ $< ${PETIGA_LIB}
	${RM} -f $<
L2Proj2D: L2Proj2D.o chkopts
	${CLINKER} -o $@ $< ${PETIGA_LIB}
	${RM} -f $<
Poisson1D: Poisson1D.o chkopts
	${CLINKER} -o $@ $< ${PETIGA_LIB}
	${RM} -f $<
Poisson2D: Poisson2D.o chkopts
	${CLINKER} -o $@ $< ${PETIGA_LIB}
	${RM} -f $<
Poisson3D: Poisson3D.o chkopts
	${CLINKER} -o $@ $< ${PETIGA_LIB}
	${RM} -f $<
Laplace: Laplace.o chkopts
	${CLINKER} -o $@ $< ${PETIGA_LIB}
	${RM} -f $<
Poisson: Poisson.o chkopts
	${CLINKER} -o $@ $< ${PETIGA_LIB}
	${RM} -f $<
PatternFormation: PatternFormation.o chkopts
	${CLINKER} -o $@ $< ${PETIGA_LIB}
	${RM} -f $<
CahnHilliard2D: CahnHilliard2D.o chkopts
	${CLINKER} -o $@ $< ${PETIGA_LIB}
	${RM} -f $<
NavierStokesVMS: NavierStokesVMS.o chkopts
	${CLINKER} -o $@ $< ${PETIGA_LIB}
	${RM} -f $<

runex0a_1:
	-@${MPIEXEC} -n 1 ./InputOutput -dim 1 -periodic           -nox -malloc_debug -malloc_dump
runex0a_4:
	-@${MPIEXEC} -n 4 ./InputOutput -dim 2 -N 17,19   -p 3,2   -nox -malloc_debug -malloc_dump
runex0a_8:
	-@${MPIEXEC} -n 8 ./InputOutput -dim 3 -N 13,11,7 -p 3,2,1 -nox -malloc_debug -malloc_dump
runex0a.rm:
	-@${RM} -f iga.dat

runex1a_1:
	-@${MPIEXEC} -n 1 ./L2Proj1D -nox -malloc_debug -malloc_dump
runex1a_4:
	-@${MPIEXEC} -n 4 ./L2Proj1D -nox -malloc_debug -malloc_dump
runex1b_1:
	-@${MPIEXEC} -n 1 ./L2Proj2D -nox -malloc_debug -malloc_dump
runex1b_4:
	-@${MPIEXEC} -n 4 ./L2Proj2D -nox -malloc_debug -malloc_dump

runex2a_1:
	-@${MPIEXEC} -n 1 ./Poisson1D -nox -malloc_debug -malloc_dump
runex2a_4:
	-@${MPIEXEC} -n 4 ./Poisson1D -nox -malloc_debug -malloc_dump
runex2b_1:
	-@${MPIEXEC} -n 1 ./Poisson2D -nox -malloc_debug -malloc_dump
runex2b_4:
	-@${MPIEXEC} -n 4 ./Poisson2D -nox -malloc_debug -malloc_dump
runex2c_1:
	-@${MPIEXEC} -n 1 ./Poisson3D -nox -malloc_debug -malloc_dump
runex2c_4:
	-@${MPIEXEC} -n 4 ./Poisson3D -nox -malloc_debug -malloc_dump
runex4_1:
	-@${MPIEXEC} -n 1 ./CahnHilliard2D -ts_max_steps 2 -malloc_debug -malloc_dump
runex4_4:
	-@${MPIEXEC} -n 2 ./CahnHilliard2D -ts_max_steps 2 -malloc_debug -malloc_dump

InputOutput := InputOutput.PETSc runex0a_1 runex0a_4 runex0a_8 runex0a.rm InputOutput.rm
L2Proj1D := L2Proj1D.PETSc runex1a_1 runex1a_4 L2Proj1D.rm
L2Proj2D := L2Proj2D.PETSc runex1b_1 runex1b_4 L2Proj2D.rm
Poisson1D := Poisson1D.PETSc runex2a_1 runex2a_4 Poisson1D.rm
Poisson2D := Poisson2D.PETSc runex2b_1 runex2b_4 Poisson2D.rm
Poisson3D := Poisson3D.PETSc runex2c_1 runex2c_4 Poisson3D.rm
CahnHilliard2D := CahnHilliard2D.PETSc runex4_1 runex4_4 CahnHilliard2D.rm

L2Proj       := $(L2Proj1D) $(L2Proj2D) $(L2Proj3D)
Poisson      := $(Poisson1D) $(Poisson2D) $(Poisson3D)
CahnHilliard := $(CahnHilliard2D) $(CahnHilliard3D)

TESTEXAMPLES_C := $(InputOutput) $(L2Proj) $(Poisson) $(CahnHilliard)
TESTEXAMPLES_F :=
TESTEXAMPLES_FORTRAN:=$(TESTEXAMPLES_F)
testexamples:
	-@${OMAKE} tree ACTION=testexamples_C PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR}
testfortran:
	-@if [ "${FC}" != "" ]; then \
            ${OMAKE} tree ACTION=testexamples_Fortran PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR}; \
          fi

include ${PETIGA_DIR}/conf/petigatest
