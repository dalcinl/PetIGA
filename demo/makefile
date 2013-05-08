TARGETS = L2Proj1D L2Proj2D \
	  Poisson1D Poisson2D Poisson3D \
	  L2Projection \
	  AdaptiveL2Projection \
	  AdvectionDiffusion \
	  Laplace \
	  Poisson \
	  Neumann \
	  Bratu \
	  PatternFormation \
	  CahnHilliard2D \
	  CahnHilliard3D \
	  NavierStokesKorteweg2D \
	  NavierStokesVMS \
	  Elasticity3D \
	  HyperElasticity \
	  PhaseFieldCrystal2D \
          BoundaryIntegral \
          Richards \
          TwoPhaseTwoComponent \
	  ShallowWater \
          ClassicalShell \
	  LoggChallenge \
          ElasticRod

ALL: ${TARGETS}
clean::
	-@${RM} ${TARGETS}

CFLAGS    = #-g3 -Wall -Wextra -Wno-unused-parameter #-Wconversion
FFLAGS    = #-g3 -Wall -Wextra -fcheck=all
CPPFLAGS  =
FPPFLAGS  =
LOCDIR    = demo/
EXAMPLESC = 
EXAMPLESF =
MANSEC    = IGA

topdir := $(shell cd .. && pwd)
PETIGA_DIR ?= $(topdir)
include ${PETIGA_DIR}/conf/petigavariables
include ${PETIGA_DIR}/conf/petigarules

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
AdaptiveL2Projection: AdaptiveL2Projection.o chkopts
	${CLINKER} -o $@ $< ${PETIGA_LIB}
	${RM} -f $<
Laplace: Laplace.o chkopts
	${CLINKER} -o $@ $< ${PETIGA_LIB}
	${RM} -f $<
Poisson: Poisson.o chkopts
	${CLINKER} -o $@ $< ${PETIGA_LIB}
	${RM} -f $<
Neumann: Neumann.o chkopts
	${CLINKER} -o $@ $< ${PETIGA_LIB}
	${RM} -f $<
LoggChallenge: LoggChallenge.o chkopts
	${CLINKER} -o $@ $< ${PETIGA_LIB}
	${RM} -f $<
Bratu: Bratu.o BratuFJ.o chkopts
	${CLINKER} -o $@ $< BratuFJ.o ${PETIGA_LIB}
	${RM} -f $< BratuFJ.o bratufj.mod
AdvectionDiffusion: AdvectionDiffusion.o chkopts
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
ElasticRod: ElasticRod.o ElasticRodFJ.o chkopts
	${CLINKER} -o $@ $< ElasticRodFJ.o ${PETIGA_LIB}
	${RM} -f $< ElasticRodFJ.o elasticrodfj.mod
Elasticity3D: Elasticity3D.o chkopts
	${CLINKER} -o $@ $< ${PETIGA_LIB}
	${RM} -f $<
HyperElasticity: HyperElasticity.o chkopts
	${CLINKER} -o $@ $< ${PETIGA_LIB}
	${RM} -f $<
PhaseFieldCrystal2D: PhaseFieldCrystal2D.o chkopts
	${CLINKER} -o $@ $< ${PETIGA_LIB}
	${RM} -f $<
BoundaryIntegral: BoundaryIntegral.o chkopts
	${CLINKER} -o $@ $< ${PETIGA_LIB}
	${RM} -f $<
Richards: Richards.o chkopts
	${CLINKER} -o $@ $< ${PETIGA_LIB}
	${RM} -f $<
TwoPhaseTwoComponent: TwoPhaseTwoComponent.o chkopts
	${CLINKER} -o $@ $< ${PETIGA_LIB}
	${RM} -f $<

OPTS=-nox -malloc_debug -malloc_dump

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
runex6a_1:
	-@${MPIEXEC} -n 1 ./Bratu ${OPTS} -iga_dim 1 -lambda 1.0
runex6a_2:
	-@${MPIEXEC} -n 2 ./Bratu ${OPTS} -iga_dim 1 -lambda 1.0 -steady false -ts_max_steps 2
runex6b_1:
	-@${MPIEXEC} -n 1 ./Bratu ${OPTS} -iga_dim 2
runex6b_4:
	-@${MPIEXEC} -n 4 ./Bratu ${OPTS} -iga_dim 2
runex6c_1:
	-@${MPIEXEC} -n 1 ./Bratu ${OPTS} -iga_dim 2 -steady false -ts_max_steps 2
runex6c_4:
	-@${MPIEXEC} -n 4 ./Bratu ${OPTS} -iga_dim 2 -steady false -ts_max_steps 2
runex6d_1:
	-@${MPIEXEC} -n 1 ./Bratu ${OPTS} -iga_dim 2 -steady -iga_collocation
runex7a_1:
	-@${MPIEXEC} -n 1 ./Neumann ${OPTS} -dim 1
runex7a_4:
	-@${MPIEXEC} -n 4 ./Neumann ${OPTS} -dim 1
runex7b_1:
	-@${MPIEXEC} -n 1 ./Neumann ${OPTS} -dim 2
runex7b_4:
	-@${MPIEXEC} -n 4 ./Neumann ${OPTS} -dim 2
runex7c_4:
	-@${MPIEXEC} -n 4 ./Neumann ${OPTS} -dim 3
runex8_1:
	-@${MPIEXEC} -n 1 ./ElasticRod ${OPTS} -ts_max_steps 10
runex8_4:
	-@${MPIEXEC} -n 4 ./ElasticRod ${OPTS} -ts_max_steps 10


L2Proj1D := L2Proj1D.PETSc runex1a_1 runex1a_4 L2Proj1D.rm
L2Proj2D := L2Proj2D.PETSc runex1b_1 runex1b_4 L2Proj2D.rm
Poisson1D := Poisson1D.PETSc runex2a_1 runex2a_4 Poisson1D.rm
Poisson2D := Poisson2D.PETSc runex2b_1 runex2b_4 Poisson2D.rm
Poisson3D := Poisson3D.PETSc runex2c_1 runex2c_4 Poisson3D.rm
Neumann := Neumann.PETSc runex7a_1 runex7a_4 runex7b_1 runex7b_4 runex7c_4 Neumann.rm
Bratu := Bratu.PETSc runex6a_1 runex6a_2 runex6b_1 runex6b_4 runex6c_1 runex6c_4 runex6d_1 Bratu.rm
CahnHilliard2D := CahnHilliard2D.PETSc runex4_1 runex4_4 CahnHilliard2D.rm
PatternFormation := PatternFormation.PETSc runex5a_1 runex5a_4 runex5b_1 runex5b_4 PatternFormation.rm
ElasticRod := ElasticRod.PETSc runex8_1 runex8_4 ElasticRod.rm

L2Proj       := $(L2Proj1D) $(L2Proj2D) $(L2Proj3D)
Poisson      := $(Poisson1D) $(Poisson2D) $(Poisson3D)
CahnHilliard := $(CahnHilliard2D) $(CahnHilliard3D)

TESTEXAMPLES_C := $(L2Proj) $(Poisson) $(Neumann) $(Bratu) $(CahnHilliard) $(PatternFormation) $(ElasticRod)
TESTEXAMPLES_F :=
TESTEXAMPLES_FORTRAN:=$(TESTEXAMPLES_F)
testexamples:
	-@${OMAKE} tree ACTION=testexamples_C PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR}
testfortran:
	-@if [ "${FC}" != "" ]; then \
	    ${OMAKE} tree ACTION=testexamples_Fortran PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR}; \
	  fi

include ${PETIGA_DIR}/conf/petigatest
