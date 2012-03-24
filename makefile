ALL: all
LOCDIR = .
DIRS   = include src docs demo test

PETIGA_DIR ?= $(CURDIR)
include ${PETIGA_DIR}/conf/petigavariables
include ${PETIGA_DIR}/conf/petigarules
include ${PETIGA_DIR}/conf/petigatest

all:
	@${OMAKE} chkpetsc_dir  PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR}
	@${OMAKE} chkpetiga_dir PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR}
	@${OMAKE} all-legacy    PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR}
all-legacy:
	-@${MKDIR} ${PETSC_ARCH}/conf
	-@${MKDIR} ${PETSC_ARCH}/include
	-@${MKDIR} ${PETSC_ARCH}/lib
	-@${OMAKE} all_build PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR} 2>&1 | tee ./${PETSC_ARCH}/conf/make.log
all_build: chk_petsc_dir chk_petiga_dir chklib_dir deletelibs deletemods build sharedlibs
.PHONY: all all-legacy all_build

#
# Check if PETSC_DIR variable specified is valid
#
chk_petsc_dir:
	@if [ ! -f ${PETSC_DIR}/include/petsc.h ]; then \
	  echo "Incorrect PETSC_DIR specified: ${PETSC_DIR}"; \
	  echo "Aborting build"; \
	  false; fi
.PHONY: chk_petsc_dir

#
# Check if PETIGA_DIR variable specified is valid
#
chk_petiga_dir:
	@if [ ! -f ${PETIGA_DIR}/include/petiga.h ]; then \
	  echo "Incorrect PETIGA_DIR specified: ${PETIGA_DIR}"; \
	  echo "Aborting build"; \
	  false; fi
.PHONY: chk_petiga_dir

#
# Build the PetIGA library
#
build:
	-@echo "========================================="
	-@echo "Building PetIGA"
	-@echo "Using PETSC_DIR=${PETSC_DIR}"
	-@echo "Using PETSC_ARCH=${PETSC_ARCH}"
	-@echo "========================================="
	-@echo "Beginning to compile PetIGA library"
#	-@echo "========================================="
	-@${OMAKE} tree ACTION=libfast PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR}
	-@${OMAKE} ranlib              PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR}
	-@echo "Completed building PetIGA library"
	-@echo "========================================="
ranlib:
	-@${RANLIB} ${PETIGA_LIB_DIR}/*.${AR_LIB_SUFFIX} > tmpf 2>&1 ; ${GREP} -v "has no symbols" tmpf; ${RM} tmpf;
sharedlibs:
	-@${OMAKE} shared_nomesg PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR} \
	           | grep -v "making shared libraries in" 
.PHONY: build ranlib sharedlibs

# Delete PetIGA library
deletelibs:
	-@${RM} -r ${PETIGA_LIB_DIR}/libpetiga*.*
deletemods:
	-@${RM} -r ${PETIGA_DIR}/${PETSC_ARCH}/include/petiga*.mod
.PHONY: deletelibs deletemods

# Clean up build
srcclean:
	-@${OMAKE} tree ACTION=clean PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR}
allclean: deletelibs deletemods srcclean
	-@${RM} -r ${PETIGA_DIR}/${PETSC_ARCH}/conf/*.log
.PHONY: srcclean allclean

# Run test examples
testexamples:
	-@echo "=========================================="
	-@echo "BEGINNING TO COMPILE AND RUN TEST EXAMPLES"
	-@echo "=========================================="
	-@${OMAKE} tree ACTION=testexamples_C PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR}
	-@echo "Completed compiling and running test examples"
	-@echo "=========================================="
.PHONY: testexamples


# Build test
test: 
	-@${OMAKE} test-build PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR} 2>&1 | tee ./${PETSC_ARCH}/conf/test.log
test-build:
	-@echo "Running test to verify correct installation"
	-@echo "Using PETIGA_DIR=${PETIGA_DIR}"
	-@echo "Using PETSC_DIR=${PETSC_DIR}"
	-@echo "Using PETSC_ARCH=${PETSC_ARCH}"
	@cd test; ${OMAKE} clean       PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR}
	@cd test; ${OMAKE} test-build  PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR}
	@cd test; ${OMAKE} clean       PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR}
	-@echo "Completed test"
.PHONY: test test-build


alletags:
	-@${PYTHON} ${PETSC_DIR}/bin/maint/generateetags.py
deleteetags:
	-@${RM} CTAGS TAGS
.PHONY: alletags deleteetags
