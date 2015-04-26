TARGETS = \
Poisson1D Poisson2D Poisson3D \
L2Projection AdaptiveL2Projection \
Laplace BoundaryIntegral \
Poisson LoggChallenge \
Neumann \
NitscheMethod \
AdvectionDiffusion \
Bratu \
PatternFormation \
CahnHilliard2D \
CahnHilliard3D \
NavierStokesKorteweg2D \
NavierStokesVMS \
Elasticity \
Elasticity3D \
HyperElasticity \
Richards \
TwoPhaseTwoComponent \
ShallowWater \
ClassicalShell \
ElasticRod

ALL: $(TARGETS)
clean::
	-@$(RM) $(TARGETS)

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
include $(PETIGA_DIR)/lib/petiga/conf/variables
include $(PETIGA_DIR)/lib/petiga/conf/rules

Poisson1D: Poisson1D.o chkopts
	$(CLINKER) -o $@ $< $(PETIGA_LIB)
	$(RM) -f $<
Poisson2D: Poisson2D.o chkopts
	$(CLINKER) -o $@ $< $(PETIGA_LIB)
	$(RM) -f $<
Poisson3D: Poisson3D.o chkopts
	$(CLINKER) -o $@ $< $(PETIGA_LIB)
	$(RM) -f $<

L2Projection: L2Projection.o chkopts
	$(CLINKER) -o $@ $< $(PETIGA_LIB)
	$(RM) -f $<
AdaptiveL2Projection: AdaptiveL2Projection.o chkopts
	$(CLINKER) -o $@ $< $(PETIGA_LIB)
	$(RM) -f $<
Laplace: Laplace.o chkopts
	$(CLINKER) -o $@ $< $(PETIGA_LIB)
	$(RM) -f $<
BoundaryIntegral: BoundaryIntegral.o chkopts
	$(CLINKER) -o $@ $< $(PETIGA_LIB)
	$(RM) -f $<
Poisson: Poisson.o chkopts
	$(CLINKER) -o $@ $< $(PETIGA_LIB)
	$(RM) -f $<
LoggChallenge: LoggChallenge.o chkopts
	$(CLINKER) -o $@ $< $(PETIGA_LIB)
	$(RM) -f $<
Neumann: Neumann.o chkopts
	$(CLINKER) -o $@ $< $(PETIGA_LIB)
	$(RM) -f $<
NitscheMethod: NitscheMethod.o chkopts
	$(CLINKER) -o $@ $< $(PETIGA_LIB)
	$(RM) -f $<
Bratu: Bratu.o BratuFJ.o chkopts
	$(CLINKER) -o $@ $< BratuFJ.o $(PETIGA_LIB)
	$(RM) -f $< BratuFJ.o bratufj.mod
AdvectionDiffusion: AdvectionDiffusion.o chkopts
	$(CLINKER) -o $@ $< $(PETIGA_LIB)
	$(RM) -f $<
PatternFormation: PatternFormation.o chkopts
	$(CLINKER) -o $@ $< $(PETIGA_LIB)
	$(RM) -f $<
CahnHilliard2D: CahnHilliard2D.o chkopts
	$(CLINKER) -o $@ $< $(PETIGA_LIB)
	$(RM) -f $<
CahnHilliard3D: CahnHilliard3D.o chkopts
	$(CLINKER) -o $@ $< $(PETIGA_LIB)
	$(RM) -f $<
NavierStokesKorteweg2D: NavierStokesKorteweg2D.o chkopts
	$(CLINKER) -o $@ $< $(PETIGA_LIB)
	$(RM) -f $<
NavierStokesVMS: NavierStokesVMS.o chkopts
	$(CLINKER) -o $@ $< $(PETIGA_LIB)
	$(RM) -f $<
ShallowWater: ShallowWater.o chkopts
	$(CLINKER) -o $@ $< $(PETIGA_LIB)
	$(RM) -f $<
ClassicalShell: ClassicalShell.o chkopts
	$(CLINKER) -o $@ $< $(PETIGA_LIB)
	$(RM) -f $<
ElasticRod: ElasticRod.o ElasticRodFJ.o chkopts
	$(CLINKER) -o $@ $< ElasticRodFJ.o $(PETIGA_LIB)
	$(RM) -f $< ElasticRodFJ.o elasticrodfj.mod
Elasticity: Elasticity.o chkopts
	$(CLINKER) -o $@ $< $(PETIGA_LIB)
	$(RM) -f $<
Elasticity3D: Elasticity3D.o chkopts
	$(CLINKER) -o $@ $< $(PETIGA_LIB)
	$(RM) -f $<
HyperElasticity: HyperElasticity.o chkopts
	$(CLINKER) -o $@ $< $(PETIGA_LIB)
	$(RM) -f $<
Richards: Richards.o chkopts
	$(CLINKER) -o $@ $< $(PETIGA_LIB)
	$(RM) -f $<
TwoPhaseTwoComponent: TwoPhaseTwoComponent.o chkopts
	$(CLINKER) -o $@ $< $(PETIGA_LIB)
	$(RM) -f $<

OPTS=-nox -malloc_debug -malloc_dump

runex1a_1:
	-@$(MPIEXEC) -n 1 ./L2Projection $(OPTS) -check_error -d 1
runex1a_4:
	-@$(MPIEXEC) -n 4 ./L2Projection $(OPTS) -check_error -d 1
runex1b_1:
	-@$(MPIEXEC) -n 1 ./L2Projection $(OPTS) -check_error -d 2
runex1b_4:
	-@$(MPIEXEC) -n 4 ./L2Projection $(OPTS) -check_error -d 2
runex1c_1:
	-@$(MPIEXEC) -n 1 ./L2Projection $(OPTS) -check_error -d 3 -p 1
runex1c_4:
	-@$(MPIEXEC) -n 4 ./L2Projection $(OPTS) -check_error -d 3 -p 1
runex2a_1:
	-@$(MPIEXEC) -n 1 ./Poisson1D $(OPTS)
runex2a_4:
	-@$(MPIEXEC) -n 4 ./Poisson1D $(OPTS)
runex2b_1:
	-@$(MPIEXEC) -n 1 ./Poisson2D $(OPTS)
runex2b_4:
	-@$(MPIEXEC) -n 4 ./Poisson2D $(OPTS)
runex2c_1:
	-@$(MPIEXEC) -n 1 ./Poisson3D $(OPTS)
runex2c_4:
	-@$(MPIEXEC) -n 4 ./Poisson3D $(OPTS)
runex3a_1:
	-@$(MPIEXEC) -n 1 ./Laplace $(OPTS) -check_error -iga_dim 1
runex3a_4:
	-@$(MPIEXEC) -n 4 ./Laplace $(OPTS) -check_error -iga_dim 1
runex3b_1:
	-@$(MPIEXEC) -n 1 ./Laplace $(OPTS) -check_error -iga_dim 2
runex3b_4:
	-@$(MPIEXEC) -n 4 ./Laplace $(OPTS) -check_error -iga_dim 2
runex3c_1:
	-@$(MPIEXEC) -n 1 ./Laplace $(OPTS) -check_error -iga_dim 3 -iga_elements 8
runex3c_4:
	-@$(MPIEXEC) -n 4 ./Laplace $(OPTS) -check_error -iga_dim 3 -iga_elements 8
runex3d_1:
	-@$(MPIEXEC) -n 1 ./Laplace $(OPTS) -check_error -iga_dim 1 -iga_collocation -iga_degree 2
runex3d_4:
	-@$(MPIEXEC) -n 4 ./Laplace $(OPTS) -check_error -iga_dim 2 -iga_collocation -iga_degree 4,6
runex4_1:
	-@$(MPIEXEC) -n 1 ./CahnHilliard2D $(OPTS) -ts_max_steps 2
runex4_4:
	-@$(MPIEXEC) -n 4 ./CahnHilliard2D $(OPTS) -ts_max_steps 2
runex5a_1:
	-@$(MPIEXEC) -n 1 ./PatternFormation $(OPTS) -ts_max_steps 2
runex5a_4:
	-@$(MPIEXEC) -n 4 ./PatternFormation $(OPTS) -ts_max_steps 2
runex5b_1:
	-@$(MPIEXEC) -n 1 ./PatternFormation $(OPTS) -ts_max_steps 2 -implicit
runex5b_4:
	-@$(MPIEXEC) -n 4 ./PatternFormation $(OPTS) -ts_max_steps 2 -implicit
runex6a_1:
	-@$(MPIEXEC) -n 1 ./Bratu $(OPTS) -iga_dim 1 -lambda 1.0
runex6a_2:
	-@$(MPIEXEC) -n 2 ./Bratu $(OPTS) -iga_dim 1 -lambda 1.0 -steady false -ts_max_steps 2
runex6b_1:
	-@$(MPIEXEC) -n 1 ./Bratu $(OPTS) -iga_dim 2
runex6b_4:
	-@$(MPIEXEC) -n 4 ./Bratu $(OPTS) -iga_dim 2
runex6c_1:
	-@$(MPIEXEC) -n 1 ./Bratu $(OPTS) -iga_dim 2 -steady false -ts_max_steps 2
runex6c_4:
	-@$(MPIEXEC) -n 4 ./Bratu $(OPTS) -iga_dim 2 -steady false -ts_max_steps 2
runex6d_1:
	-@$(MPIEXEC) -n 1 ./Bratu $(OPTS) -iga_dim 2 -steady true -iga_collocation
runex6d_2:
	-@$(MPIEXEC) -n 2 ./Bratu $(OPTS) -iga_dim 2 -steady true -iga_collocation
runex6d_4:
	-@$(MPIEXEC) -n 4 ./Bratu $(OPTS) -iga_dim 2 -steady true -iga_collocation
runex6d_6:
	-@$(MPIEXEC) -n 6 ./Bratu $(OPTS) -iga_dim 2 -steady true -iga_collocation
runex6d_8:
	-@$(MPIEXEC) -n 8 ./Bratu $(OPTS) -iga_dim 2 -steady true -iga_collocation
runex6e_1:
	-@$(MPIEXEC) -n 1 ./Bratu $(OPTS) -iga_dim 2 -iga_degree 1 -lambda 1.0 -iga_fd
	-@$(MPIEXEC) -n 1 ./Bratu $(OPTS) -iga_dim 2 -iga_degree 1 -lambda 1.0 -snes_fd_color
runex6e_4:
	-@$(MPIEXEC) -n 1 ./Bratu $(OPTS) -iga_dim 2 -iga_degree 1 -lambda 1.0 -iga_fd
	-@$(MPIEXEC) -n 4 ./Bratu $(OPTS) -iga_dim 2 -iga_degree 1 -lambda 1.0 -snes_fd_color
runex6f_1:
	-@$(MPIEXEC) -n 1 ./Bratu $(OPTS) -iga_dim 2 -iga_degree 1 -lambda 1.0 -iga_fd         -steady false -ts_max_steps 2
	-@$(MPIEXEC) -n 1 ./Bratu $(OPTS) -iga_dim 2 -iga_degree 1 -lambda 1.0 -snes_fd_color  -steady false -ts_max_steps 2
runex6f_4:
	-@$(MPIEXEC) -n 1 ./Bratu $(OPTS) -iga_dim 2 -iga_degree 1 -lambda 1.0 -iga_fd	       -steady false -ts_max_steps 2
	-@$(MPIEXEC) -n 4 ./Bratu $(OPTS) -iga_dim 2 -iga_degree 1 -lambda 1.0 -snes_fd_color  -steady false -ts_max_steps 2
runex7a_1:
	-@$(MPIEXEC) -n 1 ./Neumann $(OPTS) -check_error -iga_dim 1 -pc_type icc
runex7a_4:
	-@$(MPIEXEC) -n 4 ./Neumann $(OPTS) -check_error -iga_dim 1 -sub_pc_type icc
runex7b_1:
	-@$(MPIEXEC) -n 1 ./Neumann $(OPTS) -check_error -iga_dim 2 -pc_type icc
runex7b_4:
	-@$(MPIEXEC) -n 4 ./Neumann $(OPTS) -check_error -iga_dim 2 -sub_pc_type icc
runex7c_4:
	-@$(MPIEXEC) -n 4 ./Neumann $(OPTS) -check_error -iga_dim 3 -sub_pc_type icc
runex7d_1:
	-@$(MPIEXEC) -n 1 ./Neumann $(OPTS) -check_error -iga_dim 1 -iga_collocation -iga_degree 4 -pc_factor_shift_type nonzero
runex7d_4:
	-@$(MPIEXEC) -n 4 ./Neumann $(OPTS) -check_error -iga_dim 2 -iga_collocation -iga_degree 6
runex8_1:
	-@$(MPIEXEC) -n 1 ./ElasticRod $(OPTS) -ts_max_steps 10
runex8_4:
	-@$(MPIEXEC) -n 4 ./ElasticRod $(OPTS) -ts_max_steps 10
runex9a_1:
	-@$(MPIEXEC) -n 1 ./NitscheMethod $(OPTS) -check_error 1e-6 -iga_dim 1 -iga_degree 2 -ksp_type cg -ksp_rtol 1e-7 -pc_type none
runex9a_4:
	-@$(MPIEXEC) -n 4 ./NitscheMethod $(OPTS) -check_error 1e-6 -iga_dim 2 -iga_degree 2 -ksp_type cg -ksp_rtol 1e-7


L2Projection := \
L2Projection.PETSc \
runex1a_1 runex1a_4 \
runex1b_1 runex1b_4 \
runex1c_1 runex1c_4 \
L2Projection.rm

Laplace := \
Laplace.PETSc \
runex3a_1 runex3a_4 \
runex3b_1 runex3b_4 \
runex3c_1 runex3c_4 \
runex3d_1 runex3d_4 \
Laplace.rm

Poisson1D := Poisson1D.PETSc runex2a_1 runex2a_4 Poisson1D.rm
Poisson2D := Poisson2D.PETSc runex2b_1 runex2b_4 Poisson2D.rm
Poisson3D := Poisson3D.PETSc runex2c_1 runex2c_4 Poisson3D.rm
Poisson   := $(Poisson1D) $(Poisson2D) $(Poisson3D)

Neumann := \
Neumann.PETSc \
runex7a_1 runex7a_4 \
runex7b_1 runex7b_4 \
runex7c_4 \
runex7d_1 runex7d_4 \
Neumann.rm

NitscheMethod := \
NitscheMethod.PETSc \
runex9a_1 runex9a_4 \
NitscheMethod.rm

Bratu := \
Bratu.PETSc \
runex6a_1 runex6a_2 runex6b_1 runex6b_4 runex6c_1 runex6c_4 \
runex6d_1 runex6d_2 runex6d_4 runex6d_6 runex6d_8 \
runex6e_1 runex6e_4 runex6f_1 runex6f_4 \
Bratu.rm

CahnHilliard2D := CahnHilliard2D.PETSc runex4_1 runex4_4 CahnHilliard2D.rm
CahnHilliard3D := CahnHilliard3D.PETSc CahnHilliard3D.rm
CahnHilliard   := $(CahnHilliard2D) $(CahnHilliard3D)

PatternFormation := PatternFormation.PETSc runex5a_1 runex5a_4 runex5b_1 runex5b_4 PatternFormation.rm
ElasticRod := ElasticRod.PETSc runex8_1 runex8_4 ElasticRod.rm


TESTEXAMPLES_C := $(L2Projection) $(Laplace) $(Poisson) $(Neumann) $(NitscheMethod) $(Bratu) $(CahnHilliard) $(PatternFormation) $(ElasticRod)
TESTEXAMPLES_F :=
TESTEXAMPLES_FORTRAN:=$(TESTEXAMPLES_F)
testexamples:
	-@$(OMAKE) tree ACTION=testexamples_C PETSC_ARCH=$(PETSC_ARCH) PETSC_DIR=$(PETSC_DIR) PETIGA_DIR=$(PETIGA_DIR)
testfortran:
	-@if [ "$(FC)" != "" ]; then \
	    $(OMAKE) tree ACTION=testexamples_Fortran PETSC_ARCH=$(PETSC_ARCH) PETSC_DIR=$(PETSC_DIR) PETIGA_DIR=$(PETIGA_DIR); \
	  fi

runex-%: ; $(OMAKE) $($*)

include $(PETIGA_DIR)/lib/petiga/conf/test
