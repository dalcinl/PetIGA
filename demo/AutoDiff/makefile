# -*- mode: makefile-gmake -*-

TARGETS = \
Bratu \
CahnHilliardPrimal \
CahnHilliardMixed

ALL: $(TARGETS)

topdir := $(shell cd ../.. && pwd)
PETIGA_DIR ?= $(topdir)
include $(PETIGA_DIR)/lib/petiga/conf/variables
include $(PETIGA_DIR)/lib/petiga/conf/rules

ifdef MINIFAD
makefile.ad = makefile.minifad
else ifdef SACADO
makefile.ad = makefile.sacado
else ifdef FADBAD
makefile.ad = makefile.fadbad
else
makefile.ad = makefile.minifad
endif
$(shell $(MAKE) -f $(makefile.ad) bootstrap)
include $(makefile.ad)
PETIGA_INCLUDE += $(AD_INCLUDE)

OPTFLAGS = -Ofast
CFLAGS   = $(OPTFLAGS)
CXXFLAGS = $(OPTFLAGS)
FFLAGS   = $(OPTFLAGS)

Bratu: Bratu.o BratuC99.o BratuF90.o BratuCXX.o BratuFAD.o | chkopts
	$(CLINKER) -o $@ $^ $(PETIGA_LIB)
	$(RM) -f $^ bratuf90.mod

CahnHilliardPrimal: CahnHilliardPrimal.o CahnHilliardPrimalC99.o CahnHilliardPrimalCXX.o CahnHilliardPrimalFAD.o | chkopts
	$(CLINKER) -o $@ $^ $(PETIGA_LIB)
	$(RM) -f $^

CahnHilliardMixed: CahnHilliardMixed.o CahnHilliardMixedC99.o CahnHilliardMixedCXX.o CahnHilliardMixedFAD.o | chkopts
	$(CLINKER) -o $@ $^ $(PETIGA_LIB)
	$(RM) -f $^


DIM = 2
DEG = 2
NEL = 64

test-all: \
test-Bratu \
test-CahnHilliardPrimal \
test-CahnHilliardMixed

test-Bratu: OPTS=
test-Bratu: MONITOR=-snes_monitor
test-Bratu:
	$(MPIEXEC) -n 4 ./Bratu -dim $(DIM) -deg $(DEG) -nel $(NEL) -c99 $(OPTS) $(MONITOR)
	$(MPIEXEC) -n 4 ./Bratu -dim $(DIM) -deg $(DEG) -nel $(NEL) -f90 $(OPTS) $(MONITOR)
	$(MPIEXEC) -n 4 ./Bratu -dim $(DIM) -deg $(DEG) -nel $(NEL) -cxx $(OPTS) $(MONITOR)
	$(MPIEXEC) -n 4 ./Bratu -dim $(DIM) -deg $(DEG) -nel $(NEL) -fad $(OPTS) $(MONITOR)

test-CahnHilliardPrimal: OPTS=-ksp_pc_side right -ts_max_steps 3
test-CahnHilliardPrimal: MONITOR=-ts_monitor -snes_monitor
test-CahnHilliardPrimal:
	$(MPIEXEC) -n 4 ./CahnHilliardPrimal -dim $(DIM) -deg $(DEG) -nel $(NEL) -c99 $(OPTS) $(MONITOR)
	$(MPIEXEC) -n 4 ./CahnHilliardPrimal -dim $(DIM) -deg $(DEG) -nel $(NEL) -f90 $(OPTS) $(MONITOR)
	$(MPIEXEC) -n 4 ./CahnHilliardPrimal -dim $(DIM) -deg $(DEG) -nel $(NEL) -cxx $(OPTS) $(MONITOR)
	$(MPIEXEC) -n 4 ./CahnHilliardPrimal -dim $(DIM) -deg $(DEG) -nel $(NEL) -fad $(OPTS) $(MONITOR)

test-CahnHilliardMixed: DEG=1
test-CahnHilliardMixed: NEL=64
test-CahnHilliardMixed: OPTS=-ksp_pc_side right -ts_max_steps 3
test-CahnHilliardMixed: MONITOR=-ts_monitor -snes_monitor
test-CahnHilliardMixed:
	$(MPIEXEC) -n 4 ./CahnHilliardMixed  -dim $(DIM) -deg $(DEG) -nel $(NEL) -c99 $(OPTS) $(MONITOR)
	$(MPIEXEC) -n 4 ./CahnHilliardMixed  -dim $(DIM) -deg $(DEG) -nel $(NEL) -f90 $(OPTS) $(MONITOR)
	$(MPIEXEC) -n 4 ./CahnHilliardMixed  -dim $(DIM) -deg $(DEG) -nel $(NEL) -cxx $(OPTS) $(MONITOR)
	$(MPIEXEC) -n 4 ./CahnHilliardMixed  -dim $(DIM) -deg $(DEG) -nel $(NEL) -fad $(OPTS) $(MONITOR)


ifeq ($(shell uname -s),Linux)
CPUMASK = 0x8
TASKSET = taskset $(CPUMASK)
endif

perf-test-Bratu: NEL=128
perf-test-Bratu:
	$(TASKSET) ./Bratu -dim $(DIM) -deg $(DEG) -nel $(NEL) -c99 -log_view "ascii:Bratu-c99.log"
	$(TASKSET) ./Bratu -dim $(DIM) -deg $(DEG) -nel $(NEL) -f90 -log_view "ascii:Bratu-f90.log"
	$(TASKSET) ./Bratu -dim $(DIM) -deg $(DEG) -nel $(NEL) -cxx -log_view "ascii:Bratu-cxx.log"
	$(TASKSET) ./Bratu -dim $(DIM) -deg $(DEG) -nel $(NEL) -fad -log_view "ascii:Bratu-fad.log"
	grep IGAFormFunction Bratu-c99.log Bratu-f90.log Bratu-cxx.log Bratu-fad.log
	grep IGAFormJacobian Bratu-c99.log Bratu-f90.log Bratu-cxx.log Bratu-fad.log

perf-test-CahnHilliardPrimal: NEL=128
perf-test-CahnHilliardPrimal: OPTS=-ksp_pc_side right -ts_max_steps 3
perf-test-CahnHilliardPrimal:
	$(TASKSET) ./CahnHilliardPrimal -dim $(DIM) -deg $(DEG) -nel $(NEL) -c99 $(OPTS) -log_view "ascii:CHPrimal-c99.log"
	$(TASKSET) ./CahnHilliardPrimal -dim $(DIM) -deg $(DEG) -nel $(NEL) -cxx $(OPTS) -log_view "ascii:CHPrimal-cxx.log"
	$(TASKSET) ./CahnHilliardPrimal -dim $(DIM) -deg $(DEG) -nel $(NEL) -fad $(OPTS) -log_view "ascii:CHPrimal-fad.log"
	grep TSStep           CHPrimal-c99.log CHPrimal-cxx.log CHPrimal-fad.log
	grep IGAFormIFunction CHPrimal-c99.log CHPrimal-cxx.log CHPrimal-fad.log
	grep IGAFormIJacobian CHPrimal-c99.log CHPrimal-cxx.log CHPrimal-fad.log

perf-test-CahnHilliardMixed: NEL=128
perf-test-CahnHilliardMixed: OPTS=-ksp_pc_side right -ts_max_steps 3
perf-test-CahnHilliardMixed:
	$(TASKSET) ./CahnHilliardMixed  -dim $(DIM) -deg $(DEG) -nel $(NEL) -c99 $(OPTS) -log_view "ascii:CHMixed-c99.log"
	$(TASKSET) ./CahnHilliardMixed  -dim $(DIM) -deg $(DEG) -nel $(NEL) -cxx $(OPTS) -log_view "ascii:CHMixed-cxx.log"
	$(TASKSET) ./CahnHilliardMixed  -dim $(DIM) -deg $(DEG) -nel $(NEL) -fad $(OPTS) -log_view "ascii:CHMixed-fad.log"
	grep TSStep           CHMixed-c99.log CHMixed-cxx.log CHMixed-fad.log
	grep IGAFormIFunction CHMixed-c99.log CHMixed-cxx.log CHMixed-fad.log
	grep IGAFormIJacobian CHMixed-c99.log CHMixed-cxx.log CHMixed-fad.log

clean::
	$(RM) $(TARGETS) *.o *.mod *.log
distclean: clean
	$(RM) -r minifad FADBAD* sacado
