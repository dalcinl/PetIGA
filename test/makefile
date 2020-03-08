ALL: test

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
include $(PETIGA_DIR)/lib/petiga/conf/variables
include $(PETIGA_DIR)/lib/petiga/conf/rules
include $(PETIGA_DIR)/lib/petiga/conf/test

OPTS=-nox -malloc_debug -malloc_dump

IGACreate: IGACreate.o chkopts
	@$(CLINKER) -o $@ $< $(PETIGA_LIB)
	@$(RM) -f $<
runex1:
	$(run-test-begin)
	@$(call mpiexec,1) ./IGACreate $(OPTS) -iga_dim 1 -iga_dof 4
	@$(call mpiexec,1) ./IGACreate $(OPTS) -iga_dim 2 -iga_dof 2
	@$(call mpiexec,1) ./IGACreate $(OPTS) -iga_dim 3 -iga_dof 1
	@$(call mpiexec,2) ./IGACreate $(OPTS) -iga_dim 1 -iga_dof 8
	@$(call mpiexec,4) ./IGACreate $(OPTS) -iga_dim 2 -iga_dof 4
	@$(call mpiexec,8) ./IGACreate $(OPTS) -iga_dim 3 -iga_dof 2
	@$(call mpiexec,1) ./IGACreate $(OPTS) -iga_dim 2 -iga_dof 3 -iga_periodic 0,1 -iga_degree 2,3
	@$(call mpiexec,1) ./IGACreate $(OPTS) -iga_dim 2 -iga_dof 3 -iga_periodic 1,0 -iga_degree 2,3
	@$(call mpiexec,1) ./IGACreate $(OPTS) -iga_dim 2 -iga_dof 3 -iga_periodic 1,1 -iga_degree 2,3
	@$(call mpiexec,4) ./IGACreate $(OPTS) -iga_dim 2 -iga_dof 5 -iga_periodic 0,0,1 -iga_degree 4,3
	@$(call mpiexec,6) ./IGACreate $(OPTS) -iga_dim 2 -iga_dof 5 -iga_periodic 0,1,0 -iga_degree 4,3
	@$(call mpiexec,8) ./IGACreate $(OPTS) -iga_dim 2 -iga_dof 5 -iga_periodic 0,1,1 -iga_degree 4,3
	@$(call mpiexec,2) ./IGACreate $(OPTS) -iga_dim 2 -iga_dof 3 -iga_periodic 0,1   -iga_degree 1,2   -iga_mat_type is
	@$(call mpiexec,3) ./IGACreate $(OPTS) -iga_dim 2 -iga_dof 2 -iga_periodic 1,0   -iga_degree 1,2   -iga_mat_type is
	@$(call mpiexec,4) ./IGACreate $(OPTS) -iga_dim 3 -iga_dof 1 -iga_periodic 0,1,1 -iga_degree 1,2,1 -iga_mat_type is
	$(run-test-end)
IGACreate = IGACreate.PETSc runex1 IGACreate.rm
TESTEXAMPLES += $(IGACreate)

IGAInputOutput: IGAInputOutput.o chkopts
	@$(CLINKER) -o $@ $< $(PETIGA_LIB)
	@$(RM) -f $<
runex2:
	$(run-test-begin)
	@$(call mpiexec,1) ./IGAInputOutput $(OPTS) -iga_dim 1 -iga_periodic
	@$(call mpiexec,2) ./IGAInputOutput $(OPTS) -iga_dim 2 -iga_dof 3 -iga_periodic
	@$(call mpiexec,3) ./IGAInputOutput $(OPTS) -iga_dim 3 -iga_periodic 1,0,1
	@$(call mpiexec,4) ./IGAInputOutput $(OPTS) -iga_dim 2 -N 17,19   -p 3,2
	@$(call mpiexec,8) ./IGAInputOutput $(OPTS) -iga_dim 3 -N 13,11,7 -p 3,2,1
	@$(RM) -f iga*.dat iga*.dat.info igavec.vts
	$(run-test-end)
IGAInputOutput = IGAInputOutput.PETSc runex2 IGAInputOutput.rm
TESTEXAMPLES += $(IGAInputOutput)


IGAGeometryMap: IGAGeometryMap.o chkopts
	@$(CLINKER) -o $@ $< $(PETIGA_LIB)
	@$(RM) -f $<
runex3:
	$(run-test-begin)
	@$(call mpiexec,1) ./IGAGeometryMap $(OPTS) -dim 2
	@$(call mpiexec,1) ./IGAGeometryMap $(OPTS) -dim 3
	$(run-test-end)
IGAGeometryMap = IGAGeometryMap.PETSc runex3 IGAGeometryMap.rm
TESTEXAMPLES += $(IGAGeometryMap)


IGAFixTable: IGAFixTable.o chkopts
	@$(CLINKER) -o $@ $< $(PETIGA_LIB)
	@$(RM) -f $<
runex4:
	$(run-test-begin)
	@$(call mpiexec, 1) ./IGAFixTable $(OPTS) -iga_dim 3 -pc_type lu -check_error 1e-6 -iga_elements 1
	@$(call mpiexec, 1) ./IGAFixTable $(OPTS) -iga_dim 1 -pc_type lu -check_error 1e-6
	@$(call mpiexec, 1) ./IGAFixTable $(OPTS) -iga_dim 3 -pc_type lu -check_error 1e-6 -iga_elements 1
	@$(call mpiexec, 1) ./IGAFixTable $(OPTS) -iga_dim 2 -pc_type lu -check_error 1e-6
	@$(call mpiexec, 1) ./IGAFixTable $(OPTS) -iga_dim 3 -pc_type lu -check_error 1e-6 -iga_elements 1
	@$(call mpiexec, 1) ./IGAFixTable $(OPTS) -iga_dim 3 -pc_type lu -check_error 1e-6 -iga_elements 4
	@$(call mpiexec, 2) ./IGAFixTable $(OPTS) -iga_dim 1 -ksp_rtol 1e-7 -check_error 1e-6
	@$(call mpiexec, 3) ./IGAFixTable $(OPTS) -iga_dim 1 -ksp_rtol 1e-7 -check_error 1e-6
	@$(call mpiexec, 4) ./IGAFixTable $(OPTS) -iga_dim 2 -ksp_rtol 1e-7 -check_error 1e-6
	@$(call mpiexec, 9) ./IGAFixTable $(OPTS) -iga_dim 2 -ksp_rtol 1e-7 -check_error 1e-6
	@$(call mpiexec, 8) ./IGAFixTable $(OPTS) -iga_dim 3 -iga_elements  8,8,8 -ksp_rtol 1e-7 -check_error 1e-6
	@$(call mpiexec,12) ./IGAFixTable $(OPTS) -iga_dim 3 -iga_elements 12,8,8 -ksp_rtol 1e-7 -check_error 1e-6
	$(run-test-end)
IGAFixTable = IGAFixTable.PETSc runex4 IGAFixTable.rm
TESTEXAMPLES += $(IGAFixTable)

IGAProbe: IGAProbe.o chkopts
	@$(CLINKER) -o $@ $< $(PETIGA_LIB)
	@$(RM) -f $<
runex5:
	$(run-test-begin)
	@$(call mpiexec,1) ./IGAProbe $(OPTS) -iga_dim 1
	@$(call mpiexec,4) ./IGAProbe $(OPTS) -iga_dim 1 -collective 1
	@$(call mpiexec,4) ./IGAProbe $(OPTS) -iga_dim 1 -collective 0
	@$(call mpiexec,1) ./IGAProbe $(OPTS) -iga_dim 2
	@$(call mpiexec,4) ./IGAProbe $(OPTS) -iga_dim 2
	@$(call mpiexec,4) ./IGAProbe $(OPTS) -iga_dim 3 -iga_elements 8
	$(run-test-end)
IGAProbe = IGAProbe.PETSc runex5 IGAProbe.rm
TESTEXAMPLES += $(IGAProbe)


IGAErrNorm: IGAErrNorm.o chkopts
	@$(CLINKER) -o $@ $< $(PETIGA_LIB)
	@$(RM) -f $<
runex6:
	$(run-test-begin)
	@$(call mpiexec,1) ./IGAErrNorm $(OPTS) -iga_dim 1
	@$(call mpiexec,1) ./IGAErrNorm $(OPTS) -iga_dim 2
	@$(call mpiexec,1) ./IGAErrNorm $(OPTS) -iga_dim 3
	$(run-test-end)
IGAErrNorm = IGAErrNorm.PETSc runex6 IGAErrNorm.rm
TESTEXAMPLES += $(IGAErrNorm)


Test_SNES_2D: Test_SNES_2D.o chkopts
	@$(CLINKER) -o $@ $< $(PETIGA_LIB)
	@$(RM) -f $<
runex7:
	$(run-test-begin)
	@$(call mpiexec,1) ./Test_SNES_2D $(OPTS) -N 8 -p 1 -iga_mat_type aij
	@$(call mpiexec,1) ./Test_SNES_2D $(OPTS) -N 8 -p 1 -iga_mat_type baij
	@$(call mpiexec,1) ./Test_SNES_2D $(OPTS) -N 8 -p 1 -iga_mat_type sbaij
	@$(call mpiexec,1) ./Test_SNES_2D $(OPTS) -N 4 -p 1 -iga_mat_type dense
	@$(call mpiexec,4) ./Test_SNES_2D $(OPTS) -N 8 -p 1 -iga_mat_type aij
	@$(call mpiexec,4) ./Test_SNES_2D $(OPTS) -N 8 -p 1 -iga_mat_type baij
	@$(call mpiexec,4) ./Test_SNES_2D $(OPTS) -N 8 -p 1 -iga_mat_type sbaij
	@$(call mpiexec,4) ./Test_SNES_2D $(OPTS) -N 4 -p 1 -iga_mat_type dense
	@$(call mpiexec,1) ./Test_SNES_2D $(OPTS) -p 2
	@$(call mpiexec,4) ./Test_SNES_2D $(OPTS) -p 2
	@$(call mpiexec,1) ./Test_SNES_2D $(OPTS) -p 3
	@$(call mpiexec,4) ./Test_SNES_2D $(OPTS) -p 3
	@$(call mpiexec,1) ./Test_SNES_2D $(OPTS) -N  8 -p 2 -ksp_type cg -iga_mat_type is -is_mat_type aij
	@$(call mpiexec,1) ./Test_SNES_2D $(OPTS) -N  8 -p 2 -ksp_type cg -iga_mat_type is -is_mat_type baij
	@$(call mpiexec,1) ./Test_SNES_2D $(OPTS) -N  8 -p 2 -ksp_type cg -iga_mat_type is -is_mat_type sbaij
	@$(call mpiexec,1) ./Test_SNES_2D $(OPTS) -N  4 -p 2 -ksp_type cg -iga_mat_type is -is_mat_type dense
	@$(call mpiexec,4) ./Test_SNES_2D $(OPTS) -N 16 -p 1 -ksp_type cg -iga_mat_type is -is_mat_type aij
	@$(call mpiexec,4) ./Test_SNES_2D $(OPTS) -N 16 -p 1 -ksp_type cg -iga_mat_type is -is_mat_type baij
	@$(call mpiexec,4) ./Test_SNES_2D $(OPTS) -N 16 -p 1 -ksp_type cg -iga_mat_type is -is_mat_type sbaij
	@$(call mpiexec,4) ./Test_SNES_2D $(OPTS) -N  8 -p 1 -ksp_type cg -iga_mat_type is -is_mat_type dense
	@$(call mpiexec,1) ./Test_SNES_2D $(OPTS) -p 2 -iga_collocation
	@$(call mpiexec,4) ./Test_SNES_2D $(OPTS) -p 4 -iga_collocation
	$(run-test-end)
Test_SNES_2D = Test_SNES_2D.PETSc runex7 Test_SNES_2D.rm
TESTEXAMPLES += $(Test_SNES_2D)


ConvTest: ConvTest.o chkopts
	@$(CLINKER) -o $@ $< $(PETIGA_LIB)
	@$(RM) -f $<

.PHONY: test
test: $(TESTEXAMPLES)

.PHONY: check
check: Test_SNES_2D.PETSc
	@if [ "$(PETSC_WITH_BATCH)" != "" ]; then \
	    echo "Running with batch filesystem; to test run test/Test_SNES_2D with" ; \
	    echo "your systems batch system"; \
	  elif [ "$(MPIEXEC)" = "/bin/false" ]; then \
	    echo "*mpiexec not found*. Please run test/Test_SNES_2D manually"; \
	  elif [ -f Test_SNES_2D ]; then \
	    $(MPIEXEC) -n 1 ./Test_SNES_2D -nox -malloc_debug -malloc_dump; \
	    if [ "$(MPIEXEC)" != "$(PETSC_DIR)/bin/mpiexec.uni" ]; then \
	      $(MPIEXEC) -n 2 ./Test_SNES_2D -nox -malloc_debug -malloc_dump; \
	      $(MPIEXEC) -n 3 ./Test_SNES_2D -nox -malloc_debug -malloc_dump; \
	      $(MPIEXEC) -n 4 ./Test_SNES_2D -nox -malloc_debug -malloc_dump; \
	    fi; \
	    $(OMAKE) Test_SNES_2D.rm PETSC_ARCH=$(PETSC_ARCH) PETSC_DIR=$(PETSC_DIR) PETIGA_DIR=$(PETIGA_DIR); \
	   fi

SOURCES = $(wildcard *.c)
TARGETS = $(SOURCES:.c=)
clean::
	-@$(RM) $(TARGETS) *.o *.mod
