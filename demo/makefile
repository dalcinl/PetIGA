TARGETS = InputOutput \
          L2Proj1D L2Proj2D \
          Poisson1D Poisson2D Poisson3D \
          L2Projection \
          AdvectionDiffusion \
          Laplace \
          Poisson \
          PatternFormation \
          CahnHilliard2D \
          CahnHilliard3D \
          NavierStokesVMS \
	  Elasticity3D
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
L2Projection: L2Projection.o chkopts
	${CLINKER} -o $@ $< ${PETIGA_LIB}
	${RM} -f $<
Laplace: Laplace.o chkopts
	${CLINKER} -o $@ $< ${PETIGA_LIB}
	${RM} -f $<
AdvectionDiffusion: AdvectionDiffusion.o chkopts
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
CahnHilliard3D: CahnHilliard3D.o chkopts
	${CLINKER} -o $@ $< ${PETIGA_LIB}
	${RM} -f $<
NavierStokesKorteweg2D: NavierStokesKorteweg2D.o chkopts
	${CLINKER} -o $@ $< ${PETIGA_LIB}
	${RM} -f $<
NavierStokesVMS: NavierStokesVMS.o chkopts
	${CLINKER} -o $@ $< ${PETIGA_LIB}
	${RM} -f $<
ShallowWater: ShallowWater.o chkopts
	${CLINKER} -o $@ $< ${PETIGA_LIB}
	${RM} -f $<
ClassicalShell: ClassicalShell.o chkopts
	${CLINKER} -o $@ $< ${PETIGA_LIB}
	${RM} -f $<
Elasticity3D: Elasticity3D.o chkopts
	${CLINKER} -o $@ $< ${PETIGA_LIB}
	${RM} -f $<

OPTS=-nox -malloc_debug -malloc_dump

runex0a_1:
	-@${MPIEXEC} -n 1 ./InputOutput ${OPTS} -dim 1 -periodic
runex0a_2:
	-@${MPIEXEC} -n 2 ./InputOutput ${OPTS} -dim 2 -periodic
runex0a_3:
	-@${MPIEXEC} -n 3 ./InputOutput ${OPTS} -dim 3 -periodic 1,0,1
runex0a_4:
	-@${MPIEXEC} -n 4 ./InputOutput ${OPTS} -dim 2 -N 17,19   -p 3,2
runex0a_8:
	-@${MPIEXEC} -n 8 ./InputOutput ${OPTS} -dim 3 -N 13,11,7 -p 3,2,1
runex0a.rm:
	-@${RM} -f iga.dat iga.dat.info

runex1a_1:
	-@${MPIEXEC} -n 1 ./L2Proj1D ${OPTS}
runex1a_4:
	-@${MPIEXEC} -n 4 ./L2Proj1D ${OPTS}
runex1b_1:
	-@${MPIEXEC} -n 1 ./L2Proj2D ${OPTS}
runex1b_4:
	-@${MPIEXEC} -n 4 ./L2Proj2D ${OPTS}

runex2a_1:
	-@${MPIEXEC} -n 1 ./Poisson1D ${OPTS}
runex2a_4:
	-@${MPIEXEC} -n 4 ./Poisson1D ${OPTS}
runex2b_1:
	-@${MPIEXEC} -n 1 ./Poisson2D ${OPTS}
runex2b_4:
	-@${MPIEXEC} -n 4 ./Poisson2D ${OPTS}
runex2c_1:
	-@${MPIEXEC} -n 1 ./Poisson3D ${OPTS}
runex2c_4:
	-@${MPIEXEC} -n 4 ./Poisson3D ${OPTS}
runex4_1:
	-@${MPIEXEC} -n 1 ./CahnHilliard2D ${OPTS} -ts_max_steps 2
runex4_4:
	-@${MPIEXEC} -n 4 ./CahnHilliard2D ${OPTS} -ts_max_steps 2
runex5a_1:
	-@${MPIEXEC} -n 1 ./PatternFormation ${OPTS} -ts_max_steps 2
runex5a_4:
	-@${MPIEXEC} -n 4 ./PatternFormation ${OPTS} -ts_max_steps 2
runex5b_1:
	-@${MPIEXEC} -n 1 ./PatternFormation ${OPTS} -ts_max_steps 2 -implicit
runex5b_4:
	-@${MPIEXEC} -n 4 ./PatternFormation ${OPTS} -ts_max_steps 2 -implicit

InputOutput := InputOutput.PETSc runex0a_1 runex0a_2 runex0a_3 runex0a_4 runex0a_8 runex0a.rm InputOutput.rm
L2Proj1D := L2Proj1D.PETSc runex1a_1 runex1a_4 L2Proj1D.rm
L2Proj2D := L2Proj2D.PETSc runex1b_1 runex1b_4 L2Proj2D.rm
Poisson1D := Poisson1D.PETSc runex2a_1 runex2a_4 Poisson1D.rm
Poisson2D := Poisson2D.PETSc runex2b_1 runex2b_4 Poisson2D.rm
Poisson3D := Poisson3D.PETSc runex2c_1 runex2c_4 Poisson3D.rm
CahnHilliard2D := CahnHilliard2D.PETSc runex4_1 runex4_4 CahnHilliard2D.rm
PatternFormation := PatternFormation.PETSc runex5a_1 runex5a_4 runex5b_1 runex5b_4 PatternFormation.rm

L2Proj       := $(L2Proj1D) $(L2Proj2D) $(L2Proj3D)
Poisson      := $(Poisson1D) $(Poisson2D) $(Poisson3D)
CahnHilliard := $(CahnHilliard2D) $(CahnHilliard3D)

TESTEXAMPLES_C := $(InputOutput) $(L2Proj) $(Poisson) $(CahnHilliard) $(PatternFormation)
TESTEXAMPLES_F :=
TESTEXAMPLES_FORTRAN:=$(TESTEXAMPLES_F)
testexamples:
	-@${OMAKE} tree ACTION=testexamples_C PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR}
testfortran:
	-@if [ "${FC}" != "" ]; then \
	    ${OMAKE} tree ACTION=testexamples_Fortran PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR}; \
	  fi

include ${PETIGA_DIR}/conf/petigatest
