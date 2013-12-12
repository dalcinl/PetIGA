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

OPTS=-nox -malloc_debug -malloc_dump

Test_Eval: Test_Eval.o chkopts
	${CLINKER} -o $@ $< ${PETIGA_LIB}
	${RM} -f $<

IGACreate: IGACreate.o chkopts
	${CLINKER} -o $@ $< ${PETIGA_LIB}
	${RM} -f $<
runex1a_seq:
	-@${MPIEXEC} -n 1 ./IGACreate ${OPTS} -iga_dim 1 -iga_dof 4
	-@${MPIEXEC} -n 1 ./IGACreate ${OPTS} -iga_dim 2 -iga_dof 2
	-@${MPIEXEC} -n 1 ./IGACreate ${OPTS} -iga_dim 3 -iga_dof 1
runex1a_mpi:
	-@${MPIEXEC} -n 2 ./IGACreate ${OPTS} -iga_dim 1 -iga_dof 8
	-@${MPIEXEC} -n 4 ./IGACreate ${OPTS} -iga_dim 2 -iga_dof 4
	-@${MPIEXEC} -n 8 ./IGACreate ${OPTS} -iga_dim 3 -iga_dof 2
runex1b_seq:
	-@${MPIEXEC} -n 1 ./IGACreate ${OPTS} -iga_dim 2 -iga_dof 3 -iga_periodic 0,1 -iga_degree 2,3
	-@${MPIEXEC} -n 1 ./IGACreate ${OPTS} -iga_dim 2 -iga_dof 3 -iga_periodic 1,0 -iga_degree 2,3
	-@${MPIEXEC} -n 1 ./IGACreate ${OPTS} -iga_dim 2 -iga_dof 3 -iga_periodic 1,1 -iga_degree 2,3
runex1b_mpi:
	-@${MPIEXEC} -n 4 ./IGACreate ${OPTS} -iga_dim 2 -iga_dof 5 -iga_periodic 0,0,1 -iga_degree 4,3
	-@${MPIEXEC} -n 6 ./IGACreate ${OPTS} -iga_dim 2 -iga_dof 5 -iga_periodic 0,1,0 -iga_degree 4,3
	-@${MPIEXEC} -n 8 ./IGACreate ${OPTS} -iga_dim 2 -iga_dof 5 -iga_periodic 0,1,1 -iga_degree 4,3
IGACreate = IGACreate.PETSc \
	    runex1a_seq runex1a_mpi \
	    runex1b_seq runex1b_mpi \
	    IGACreate.rm


IGAInputOutput: IGAInputOutput.o chkopts
	${CLINKER} -o $@ $< ${PETIGA_LIB}
	${RM} -f $<
runex2a_1:
	-@${MPIEXEC} -n 1 ./IGAInputOutput ${OPTS} -iga_dim 1 -periodic
runex2a_2:
	-@${MPIEXEC} -n 2 ./IGAInputOutput ${OPTS} -iga_dim 2 -iga_dof 3 -periodic
runex2a_3:
	-@${MPIEXEC} -n 3 ./IGAInputOutput ${OPTS} -iga_dim 3 -periodic 1,0,1
runex2a_4:
	-@${MPIEXEC} -n 4 ./IGAInputOutput ${OPTS} -iga_dim 2 -N 17,19   -p 3,2
runex2a_8:
	-@${MPIEXEC} -n 8 ./IGAInputOutput ${OPTS} -iga_dim 3 -N 13,11,7 -p 3,2,1
runex2a.rm:
	-@${RM} -f iga*.dat iga*.dat.info
IGAInputOutput = IGAInputOutput.PETSc \
		 runex2a_1 runex2a_2 runex2a_3 runex2a_4 runex2a_8 \
		 runex2a.rm IGAInputOutput.rm

FixTable: FixTable.o chkopts
	${CLINKER} -o $@ $< ${PETIGA_LIB}
	${RM} -f $<
runex4a_1:
	-@${MPIEXEC} -n 1 ./FixTable ${OPTS} -iga_dim 3 -pc_type lu -check_error 1e-6 -iga_elements 1
	-@${MPIEXEC} -n 1 ./FixTable ${OPTS} -iga_dim 1 -pc_type lu -check_error 1e-6
runex4a_2:
	-@${MPIEXEC} -n 1 ./FixTable ${OPTS} -iga_dim 3 -pc_type lu -check_error 1e-6 -iga_elements 1
	-@${MPIEXEC} -n 1 ./FixTable ${OPTS} -iga_dim 2 -pc_type lu -check_error 1e-6
runex4a_3:
	-@${MPIEXEC} -n 1 ./FixTable ${OPTS} -iga_dim 3 -pc_type lu -check_error 1e-6 -iga_elements 1
	-@${MPIEXEC} -n 1 ./FixTable ${OPTS} -iga_dim 3 -pc_type lu -check_error 1e-6 -iga_elements 4
runex4b_1:
	-@${MPIEXEC} -n 2 ./FixTable ${OPTS} -iga_dim 1 -ksp_rtol 1e-7 -check_error 1e-6
	-@${MPIEXEC} -n 3 ./FixTable ${OPTS} -iga_dim 1 -ksp_rtol 1e-7 -check_error 1e-6
runex4b_2:
	-@${MPIEXEC} -n 4 ./FixTable ${OPTS} -iga_dim 2 -ksp_rtol 1e-7 -check_error 1e-6
	-@${MPIEXEC} -n 9 ./FixTable ${OPTS} -iga_dim 2 -ksp_rtol 1e-7 -check_error 1e-6
runex4b_3:
	-@${MPIEXEC} -n  8 ./FixTable ${OPTS} -iga_dim 3 -iga_elements  8,8,8 -ksp_rtol 1e-7 -check_error 1e-6
	-@${MPIEXEC} -n 12 ./FixTable ${OPTS} -iga_dim 3 -iga_elements 12,8,8 -ksp_rtol 1e-7 -check_error 1e-6
FixTable = FixTable.PETSc \
           runex4a_1 runex4a_2 runex4a_3 \
           runex4b_1 runex4b_2 runex4b_3 \
           FixTable.rm

GeometryMap: GeometryMap.o chkopts
	${CLINKER} -o $@ $< ${PETIGA_LIB}
	${RM} -f $<
runex3a_2:
	-@${MPIEXEC} -n 1 ./GeometryMap ${OPTS} -dim 2
runex3a_3:
	-@${MPIEXEC} -n 1 ./GeometryMap ${OPTS} -dim 3
GeometryMap = GeometryMap.PETSc \
	      runex3a_2 runex3a_3 \
	      GeometryMap.rm


Test_SNES_2D: Test_SNES_2D.o chkopts
	${CLINKER} -o $@ $< ${PETIGA_LIB}
	${RM} -f $<
runex0a_1:
	-@${MPIEXEC} -n 1 ./Test_SNES_2D ${OPTS} -N 8 -p 1 -iga_mat_type aij
	-@${MPIEXEC} -n 1 ./Test_SNES_2D ${OPTS} -N 8 -p 1 -iga_mat_type baij
	-@${MPIEXEC} -n 1 ./Test_SNES_2D ${OPTS} -N 8 -p 1 -iga_mat_type sbaij
	-@${MPIEXEC} -n 1 ./Test_SNES_2D ${OPTS} -N 4 -p 1 -iga_mat_type dense
runex0a_4:
	-@${MPIEXEC} -n 4 ./Test_SNES_2D ${OPTS} -N 8 -p 1 -iga_mat_type aij
	-@${MPIEXEC} -n 4 ./Test_SNES_2D ${OPTS} -N 8 -p 1 -iga_mat_type baij
	-@${MPIEXEC} -n 4 ./Test_SNES_2D ${OPTS} -N 8 -p 1 -iga_mat_type sbaij
	-@${MPIEXEC} -n 4 ./Test_SNES_2D ${OPTS} -N 4 -p 1 -iga_mat_type dense
runex0b_1:
	-@${MPIEXEC} -n 1 ./Test_SNES_2D ${OPTS} -p 2
runex0b_4:
	-@${MPIEXEC} -n 4 ./Test_SNES_2D ${OPTS} -p 2
runex0c_1:
	-@${MPIEXEC} -n 1 ./Test_SNES_2D ${OPTS} -p 3
runex0c_4:
	-@${MPIEXEC} -n 4 ./Test_SNES_2D ${OPTS} -p 3
runex0d_1:
	-@${MPIEXEC} -n 1 ./Test_SNES_2D ${OPTS} -N  8 -p 2 -ksp_type cg -iga_mat_type is -is_mat_type aij
	-@${MPIEXEC} -n 1 ./Test_SNES_2D ${OPTS} -N  8 -p 2 -ksp_type cg -iga_mat_type is -is_mat_type baij
	-@${MPIEXEC} -n 1 ./Test_SNES_2D ${OPTS} -N  8 -p 2 -ksp_type cg -iga_mat_type is -is_mat_type sbaij
	-@${MPIEXEC} -n 1 ./Test_SNES_2D ${OPTS} -N  4 -p 2 -ksp_type cg -iga_mat_type is -is_mat_type dense
runex0d_4:
	-@${MPIEXEC} -n 4 ./Test_SNES_2D ${OPTS} -N 16 -p 1 -ksp_type cg -iga_mat_type is -is_mat_type aij
	-@${MPIEXEC} -n 4 ./Test_SNES_2D ${OPTS} -N 16 -p 1 -ksp_type cg -iga_mat_type is -is_mat_type baij
	-@${MPIEXEC} -n 4 ./Test_SNES_2D ${OPTS} -N 16 -p 1 -ksp_type cg -iga_mat_type is -is_mat_type sbaij
	-@${MPIEXEC} -n 4 ./Test_SNES_2D ${OPTS} -N  8 -p 1 -ksp_type cg -iga_mat_type is -is_mat_type dense
runex0e_1:
	-@${MPIEXEC} -n 1 ./Test_SNES_2D ${OPTS} -p 2 -iga_collocation
runex0e_4:
	-@${MPIEXEC} -n 4 ./Test_SNES_2D ${OPTS} -p 4 -iga_collocation

Test_SNES_2D = Test_SNES_2D.PETSc  \
	       runex0a_1 runex0a_4 \
	       runex0b_1 runex0b_4 \
	       runex0c_1 runex0c_4 \
	       runex0d_1 runex0d_4 \
	       runex0e_1 runex0e_4 \
	       Test_SNES_2D.rm


Oscillator: Oscillator.o chkopts
	${CLINKER} -o $@ $< ${PETIGA_LIB}
	${RM} -f $<
runex20a:
	-@./Oscillator ${OPTS} -ts_max_steps 10
runex20b:
	-@./Oscillator ${OPTS} -ts_max_steps 10 -ts_alpha_radius 0.8
runex20c:
	-@./Oscillator ${OPTS} -damping 0.1
Oscillator = Oscillator.PETSc \
	     runex20a runex20b runex20c \
	     Oscillator.rm


ConvTest: ConvTest.o chkopts
	${CLINKER} -o $@ $< ${PETIGA_LIB}
	${RM} -f $<

TESTEXAMPLES_C = $(IGACreate) \
		 $(IGAInputOutput) \
		 $(FixTable) \
		 $(GeometryMap) \
		 $(Test_SNES_2D) \
		 $(Oscillator)
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
	      ${MPIEXEC} -n 9 ./Test_SNES_2D -nox -malloc_debug -malloc_dump; \
	    fi; \
	    ${OMAKE} Test_SNES_2D.rm PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR}; \
	   fi
.PHONY:test-build

include ${PETIGA_DIR}/conf/petigatest
