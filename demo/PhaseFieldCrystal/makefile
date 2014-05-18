TARGETS = \
Mixed222Stable

ALL: ${TARGETS}
clean::
	-@${RM} ${TARGETS}

CFLAGS    =
FFLAGS    =
CPPFLAGS  =
FPPFLAGS  =
LOCDIR    = demo/PhaseFieldCrystal/
EXAMPLESC =
EXAMPLESF =
MANSEC    = IGA

topdir := $(shell cd ../.. && pwd)
PETIGA_DIR ?= $(topdir)
include ${PETIGA_DIR}/conf/petigavariables
include ${PETIGA_DIR}/conf/petigarules

Mixed222Stable: Mixed222Stable.o chkopts
	${CLINKER} -o $@ $< ${PETIGA_LIB}
	${RM} -f $<

include ${PETIGA_DIR}/conf/petigatest
