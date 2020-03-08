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

runex1:
	$(run-test-begin)
	@$(call mpiexec,1) ./L2Projection $(OPTS) -check_error -d 1
	@$(call mpiexec,4) ./L2Projection $(OPTS) -check_error -d 1
	@$(call mpiexec,1) ./L2Projection $(OPTS) -check_error -d 2
	@$(call mpiexec,4) ./L2Projection $(OPTS) -check_error -d 2
	@$(call mpiexec,1) ./L2Projection $(OPTS) -check_error -d 3 -p 1
	@$(call mpiexec,4) ./L2Projection $(OPTS) -check_error -d 3 -p 1
	$(run-test-end)

runex2-1d:
	$(run-test-begin)
	@$(call mpiexec,1) ./Poisson1D $(OPTS)
	@$(call mpiexec,4) ./Poisson1D $(OPTS)
	$(run-test-end)
runex2-2d:
	$(run-test-begin)
	@$(call mpiexec,1) ./Poisson2D $(OPTS)
	@$(call mpiexec,4) ./Poisson2D $(OPTS)
	$(run-test-end)
runex2-3d:
	$(run-test-begin)
	@$(call mpiexec,1) ./Poisson3D $(OPTS)
	@$(call mpiexec,4) ./Poisson3D $(OPTS)
	$(run-test-end)

runex3:
	$(run-test-begin)
	@$(call mpiexec,1) ./Laplace $(OPTS) -check_error -iga_dim 1
	@$(call mpiexec,4) ./Laplace $(OPTS) -check_error -iga_dim 1
	@$(call mpiexec,1) ./Laplace $(OPTS) -check_error -iga_dim 2
	@$(call mpiexec,4) ./Laplace $(OPTS) -check_error -iga_dim 2
	@$(call mpiexec,1) ./Laplace $(OPTS) -check_error -iga_dim 3 -iga_elements 8
	@$(call mpiexec,4) ./Laplace $(OPTS) -check_error -iga_dim 3 -iga_elements 8
	@$(call mpiexec,1) ./Laplace $(OPTS) -check_error -iga_dim 1 -iga_collocation -iga_degree 2
	@$(call mpiexec,4) ./Laplace $(OPTS) -check_error -iga_dim 2 -iga_collocation -iga_degree 4,6
	$(run-test-end)

runex4:
	$(run-test-begin)
	@$(call mpiexec,1) ./CahnHilliard2D $(OPTS) -ts_max_steps 2
	@$(call mpiexec,4) ./CahnHilliard2D $(OPTS) -ts_max_steps 2
	$(run-test-end)

runex5:
	$(run-test-begin)
	@$(call mpiexec,1) ./PatternFormation $(OPTS) -ts_max_steps 2
	@$(call mpiexec,4) ./PatternFormation $(OPTS) -ts_max_steps 2
	@$(call mpiexec,1) ./PatternFormation $(OPTS) -ts_max_steps 2 -implicit
	@$(call mpiexec,4) ./PatternFormation $(OPTS) -ts_max_steps 2 -implicit
	$(run-test-end)

runex6:
	$(run-test-begin)
	@$(call mpiexec,1) ./Bratu $(OPTS) -iga_dim 1 -lambda 1.0
	@$(call mpiexec,2) ./Bratu $(OPTS) -iga_dim 1 -lambda 1.0 -steady false -ts_max_steps 2
	@$(call mpiexec,1) ./Bratu $(OPTS) -iga_dim 2
	@$(call mpiexec,4) ./Bratu $(OPTS) -iga_dim 2
	@$(call mpiexec,1) ./Bratu $(OPTS) -iga_dim 2 -steady false -ts_max_steps 2
	@$(call mpiexec,4) ./Bratu $(OPTS) -iga_dim 2 -steady false -ts_max_steps 2
	@$(call mpiexec,1) ./Bratu $(OPTS) -iga_dim 2 -steady true -iga_collocation
	@$(call mpiexec,2) ./Bratu $(OPTS) -iga_dim 2 -steady true -iga_collocation
	@$(call mpiexec,4) ./Bratu $(OPTS) -iga_dim 2 -steady true -iga_collocation
	@$(call mpiexec,6) ./Bratu $(OPTS) -iga_dim 2 -steady true -iga_collocation
	@$(call mpiexec,8) ./Bratu $(OPTS) -iga_dim 2 -steady true -iga_collocation
	@$(call mpiexec,1) ./Bratu $(OPTS) -iga_dim 2 -iga_degree 1 -lambda 1.0 -iga_fd
	@$(call mpiexec,1) ./Bratu $(OPTS) -iga_dim 2 -iga_degree 1 -lambda 1.0 -snes_fd_color
	@$(call mpiexec,1) ./Bratu $(OPTS) -iga_dim 2 -iga_degree 1 -lambda 1.0 -iga_fd
	@$(call mpiexec,4) ./Bratu $(OPTS) -iga_dim 2 -iga_degree 1 -lambda 1.0 -snes_fd_color
	@$(call mpiexec,1) ./Bratu $(OPTS) -iga_dim 2 -iga_degree 1 -lambda 1.0 -iga_fd         -steady false -ts_max_steps 2
	@$(call mpiexec,1) ./Bratu $(OPTS) -iga_dim 2 -iga_degree 1 -lambda 1.0 -snes_fd_color  -steady false -ts_max_steps 2
	@$(call mpiexec,1) ./Bratu $(OPTS) -iga_dim 2 -iga_degree 1 -lambda 1.0 -iga_fd	       -steady false -ts_max_steps 2
	@$(call mpiexec,4) ./Bratu $(OPTS) -iga_dim 2 -iga_degree 1 -lambda 1.0 -snes_fd_color  -steady false -ts_max_steps 2
	$(run-test-end)

runex7:
	$(run-test-begin)
	@$(call mpiexec,1) ./Neumann $(OPTS) -check_error -iga_dim 1 -pc_type icc
	@$(call mpiexec,4) ./Neumann $(OPTS) -check_error -iga_dim 1 -sub_pc_type icc
	@$(call mpiexec,1) ./Neumann $(OPTS) -check_error -iga_dim 2 -pc_type icc
	@$(call mpiexec,4) ./Neumann $(OPTS) -check_error -iga_dim 2 -sub_pc_type icc
	@$(call mpiexec,4) ./Neumann $(OPTS) -check_error -iga_dim 3 -sub_pc_type icc
	@$(call mpiexec,1) ./Neumann $(OPTS) -check_error -iga_dim 1 -iga_collocation -iga_degree 4 -pc_factor_shift_type nonzero
	@$(call mpiexec,4) ./Neumann $(OPTS) -check_error -iga_dim 2 -iga_collocation -iga_degree 6
	$(run-test-end)

runex8:
	$(run-test-begin)
	@$(call mpiexec,1) ./ElasticRod $(OPTS) -ts_max_steps 10
	@$(call mpiexec,4) ./ElasticRod $(OPTS) -ts_max_steps 10
	$(run-test-end)

runex9:
	$(run-test-begin)
	@$(call mpiexec,1) ./NitscheMethod $(OPTS) -check_error 1e-6 -iga_dim 1 -iga_degree 2 -ksp_type cg -ksp_rtol 1e-7 -pc_type none
	@$(call mpiexec,4) ./NitscheMethod $(OPTS) -check_error 1e-6 -iga_dim 2 -iga_degree 2 -ksp_type cg -ksp_rtol 1e-7
	$(run-test-end)


L2Projection := L2Projection.PETSc runex1 L2Projection.rm
Laplace := Laplace.PETSc runex3 Laplace.rm
Poisson1D := Poisson1D.PETSc runex2-1d Poisson1D.rm
Poisson2D := Poisson2D.PETSc runex2-2d Poisson2D.rm
Poisson3D := Poisson3D.PETSc runex2-3d Poisson3D.rm
Poisson := $(Poisson1D) $(Poisson2D) $(Poisson3D)
Neumann := Neumann.PETSc runex7 Neumann.rm
NitscheMethod := NitscheMethod.PETSc runex9 NitscheMethod.rm
Bratu := Bratu.PETSc runex6 Bratu.rm
CahnHilliard2D := CahnHilliard2D.PETSc runex4 CahnHilliard2D.rm
CahnHilliard3D := CahnHilliard3D.PETSc CahnHilliard3D.rm
CahnHilliard   := $(CahnHilliard2D) $(CahnHilliard3D)
PatternFormation := PatternFormation.PETSc runex5 PatternFormation.rm
ElasticRod := ElasticRod.PETSc runex8 ElasticRod.rm

TESTEXAMPLES := $(L2Projection) $(Laplace) $(Poisson) $(Neumann) $(NitscheMethod) $(Bratu) $(CahnHilliard) $(PatternFormation) $(ElasticRod)

.PHONY: test
test: $(TESTEXAMPLES)


runex-%: ; $(OMAKE) $($*)

include $(PETIGA_DIR)/lib/petiga/conf/test
