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
.PHONY: all

#
# Legacy build
#
all-legacy:
	@${MKDIR} ${PETSC_ARCH}/conf ${PETSC_ARCH}/include ${PETSC_ARCH}/lib
	@${OMAKE} all_build PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR} 2>&1 | tee ./${PETSC_ARCH}/conf/make.log
	@${MV} -f ${PETIGA_DIR}/src/petiga*.mod ${PETIGA_DIR}/${PETSC_ARCH}/include
all_build: chk_petsc_dir chk_petiga_dir chklib_dir deletelibs deletemods build
.PHONY: all-legacy all_build

#
# CMake build
#
cmake_cc=-DCMAKE_C_COMPILER:FILEPATH=${PCC} -DCMAKE_C_FLAGS:STRING='${PCC_FLAGS} ${CFLAGS} ${PETSCFLAGS} ${CPP_FLAGS} ${CPPFLAGS}'
cmake_fc=-DCMAKE_Fortran_COMPILER:FILEPATH=${FC} -DCMAKE_Fortran_FLAGS:STRING='${FC_FLAGS} ${FFLAGS} ${PETSCFLAGS} ${FPP_FLAGS} ${FPPFLAGS}'
${PETIGA_DIR}/${PETSC_ARCH}/conf:
	@${MKDIR} ${PETIGA_DIR}/${PETSC_ARCH}/conf
${PETIGA_DIR}/${PETSC_ARCH}/CMakeCache.txt: ${PETIGA_DIR}/${PETSC_ARCH}/conf
	@${RM} -r ${PETIGA_DIR}/${PETSC_ARCH}/CMakeFiles
	@cd ${PETIGA_DIR}/${PETSC_ARCH} && ${CMAKE} ${PETIGA_DIR} ${cmake_cc} ${cmake_fc} 2>&1 > ${PETIGA_DIR}/${PETSC_ARCH}/conf/cmake.log
cmake-boot: ${PETIGA_DIR}/${PETSC_ARCH}/CMakeCache.txt
cmake-build: cmake-boot
	@cd ${PETIGA_DIR}/${PETSC_ARCH} && ${OMAKE} -j ${MAKE_NP} 2>&1
all-cmake: ${PETIGA_DIR}/${PETSC_ARCH}/conf
	@${OMAKE} cmake-build PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR} 2>&1 | tee ./${PETSC_ARCH}/conf/make.log
.PHONY: cmake-boot cmake-build all-cmake

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
	-@echo "============================================="
	-@echo "Building PetIGA"
	-@echo "Using PETIGA_DIR=${PETIGA_DIR}"
	-@echo "Using PETSC_DIR=${PETSC_DIR}"
	-@echo "Using PETSC_ARCH=${PETSC_ARCH}"
	-@echo "============================================="
	-@echo "Beginning to build PetIGA library"
	-@${OMAKE} compile PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR}
	-@${OMAKE} ranlib  PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR}
	-@${OMAKE} shlibs  PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR}
	-@echo "Completed building PetIGA library"
	-@echo "============================================="
compile:
	-@${OMAKE} tree ACTION=libfast PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR}
ranlib:
	-@echo "building libpetiga.${AR_LIB_SUFFIX}"
	-@${RANLIB} ${PETIGA_LIB_DIR}/*.${AR_LIB_SUFFIX} > tmpf 2>&1 ; ${GREP} -v "has no symbols" tmpf; ${RM} tmpf;
shlibs:
	-@${OMAKE} shared_nomesg PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR} \
		   | (${GREP} -vE "making shared libraries in" || true) \
		   | (${GREP} -vE "==========================" || true)
.PHONY: build compile ranlib shlibs

# Delete PetIGA library
deletemods:
	-@${RM} -r ${PETIGA_DIR}/${PETSC_ARCH}/include/petiga*.mod
deletestaticlibs:
	-@${RM} -r ${PETIGA_LIB_DIR}/libpetiga*.${AR_LIB_SUFFIX}
deletesharedlibs:
	-@${RM} -r ${PETIGA_LIB_DIR}/libpetiga*.${SL_LINKER_SUFFIX}*
deletelibs:
	-@${RM} -r ${PETIGA_LIB_DIR}/libpetiga*.*
.PHONY: deletemods deletestaticlibs deletesharedlibs deletelibs

# Clean up build
srcclean:
	-@${OMAKE} tree ACTION=clean PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR}
allclean: deletelibs deletemods srcclean
	-@${RM} -r ${PETIGA_DIR}/${PETSC_ARCH}/conf/*.log
.PHONY: srcclean allclean

# Run test examples
testexamples:
	-@echo "============================================="
	-@echo "Beginning to compile and run test examples"
	-@echo "============================================="
	-@${OMAKE} tree ACTION=testexamples_C PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR}
	-@echo "Completed compiling and running test examples"
	-@echo "============================================="
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

doc:
	@if [ ! -d docs/html ]; then \
	  ${MKDIR} docs/html; \
	fi
	-@${RM} docs/html/*.html
	@${PETSC_DIR}/${PETSC_ARCH}/bin/doctext -mpath docs/html -html src/*.c
	@#! /bin/sh
	@echo '<TITLE>PetIGA Documentation</TITLE>' > docs/html/index.html
	@echo '<H1>PetIGA Documentation</H1>' >> docs/html/index.html
	@echo '<MENU>' >> docs/html/index.html
	@ls -1 docs/html | grep .html | grep -v index.html | sed -e 's%^\(.*\).html$$%<LI><A HREF="\1.html">\1</A>%g' >> docs/html/index.html
	@echo '</MENU>' >> docs/html/index.html

alletags:
	-@${PYTHON} ${PETSC_DIR}/bin/maint/generateetags.py
deleteetags:
	-@${RM} CTAGS TAGS
.PHONY: alletags deleteetags
