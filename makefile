ALL: all
LOCDIR = .
DIRS   = include src docs test demo

PETIGA_DIR ?= $(CURDIR)
include ${PETIGA_DIR}/lib/petiga/conf/variables
include ${PETIGA_DIR}/lib/petiga/conf/rules
include ${PETIGA_DIR}/lib/petiga/conf/test

all:
	@if [ "${MAKE_IS_GNUMAKE}" != "" ]; then \
	${OMAKE} all-gmake  PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR} PETIGA_ARCH=${PETIGA_ARCH}; \
	elif [ "${PETSC_BUILD_USING_CMAKE}" != "" ]; then \
	${OMAKE} all-cmake  PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR} PETIGA_ARCH=${PETIGA_ARCH}; else \
	${OMAKE} all-legacy PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR} PETIGA_ARCH=${PETIGA_ARCH}; fi;
.PHONY: all


${PETIGA_DIR}/${PETIGA_ARCH}/include:
	@${MKDIR} ${PETIGA_DIR}/${PETIGA_ARCH}/include
${PETIGA_DIR}/${PETIGA_ARCH}/lib:
	@${MKDIR} ${PETIGA_DIR}/${PETIGA_ARCH}/lib
${PETIGA_DIR}/${PETIGA_ARCH}/log:
	@${MKDIR} ${PETIGA_DIR}/${PETIGA_ARCH}/log
arch-tree: ${PETIGA_DIR}/${PETIGA_ARCH}/include \
           ${PETIGA_DIR}/${PETIGA_ARCH}/lib \
	   ${PETIGA_DIR}/${PETIGA_ARCH}/log
.PHONY: arch-tree


#
# GNU Make build
#
ifndef MAKE_IS_GNUMAKE
MAKE_IS_GNUMAKE = $(if $(findstring GNU Make,$(shell $(OMAKE) --version 2>/dev/null)),1,)
endif
ifdef OMAKE_PRINTDIR
GMAKE = ${OMAKE_PRINTDIR}
else
GMAKE = ${OMAKE}
endif
gmake-build:
	@cd ${PETIGA_DIR} && ${GMAKE} -f gmakefile -j ${MAKE_NP}
gmake-clean:
	@cd ${PETIGA_DIR} && ${GMAKE} -f gmakefile clean
all-gmake: chk_petsc_dir chk_petiga_dir arch-tree
	-@echo "=================================================="
	-@echo "Building PetIGA (GNU Make - ${MAKE_NP} build jobs)"
	-@echo "Using PETIGA_DIR=${PETIGA_DIR}"
	-@echo "Using PETIGA_ARCH=${PETIGA_ARCH}"
	-@echo "Using PETSC_DIR=${PETSC_DIR}"
	-@echo "Using PETSC_ARCH=${PETSC_ARCH}"
	-@echo "=================================================="
	@${GMAKE} gmake-build PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR} PETIGA_ARCH=${PETIGA_ARCH} 2>&1 | tee ./${PETIGA_ARCH}/log/make.log
	-@echo "=================================================="
.PHONY: gmake-build gmake-clean all-gmake


#
# CMake build
#
ifeq (${PETSC_LANGUAGE},CXXONLY)
cmake_cc_clang=-DPETSC_CLANGUAGE_Cxx:STRING='YES'
cmake_cc_path =-DCMAKE_CXX_COMPILER:FILEPATH=${CXX}
cmake_cc_flags=-DCMAKE_CXX_FLAGS:STRING='${PCC_FLAGS} ${CFLAGS} ${CCPPFLAGS}'
else
cmake_cc_clang=-DPETSC_CLANGUAGE_Cxx:STRING='NO'
cmake_cc_path =-DCMAKE_C_COMPILER:FILEPATH=${CC}
cmake_cc_flags=-DCMAKE_C_FLAGS:STRING='${PCC_FLAGS} ${CFLAGS} ${CCPPFLAGS}'
endif
ifneq (${FC},)
cmake_fc_path =-DCMAKE_Fortran_COMPILER:FILEPATH=${FC}
endif
ifneq (${FC_FLAGS},)
cmake_fc_flags=-DCMAKE_Fortran_FLAGS:STRING='${FC_FLAGS} ${FFLAGS} ${FCPPFLAGS}'
endif
cmake_cc=${cmake_cc_path} ${cmake_cc_flags} ${cmake_cc_clang}
cmake_fc=${cmake_fc_path} ${cmake_fc_flags}
${PETIGA_DIR}/${PETIGA_ARCH}/CMakeCache.txt: CMakeLists.txt
	-@${RM} -r ${PETIGA_DIR}/${PETIGA_ARCH}/CMakeCache.txt
	-@${RM} -r ${PETIGA_DIR}/${PETIGA_ARCH}/CMakeFiles
	-@${RM} -r ${PETIGA_DIR}/${PETIGA_ARCH}/Makefile
	-@${RM} -r ${PETIGA_DIR}/${PETIGA_ARCH}/cmake_install.cmake
	@${MKDIR} ${PETIGA_DIR}/${PETIGA_ARCH}
	@cd ${PETIGA_DIR}/${PETIGA_ARCH} && ${CMAKE} ${PETIGA_DIR} ${cmake_cc} ${cmake_fc}
cmake-boot: ${PETIGA_DIR}/${PETIGA_ARCH}/CMakeCache.txt
cmake-down:
	-@${RM} -r ${PETIGA_DIR}/${PETIGA_ARCH}/CMakeCache.txt
	-@${RM} -r ${PETIGA_DIR}/${PETIGA_ARCH}/CMakeFiles
	-@${RM} -r ${PETIGA_DIR}/${PETIGA_ARCH}/Makefile
	-@${RM} -r ${PETIGA_DIR}/${PETIGA_ARCH}/cmake_install.cmake
cmake-build: cmake-boot
	@cd ${PETIGA_DIR}/${PETIGA_ARCH} && ${OMAKE} -j ${MAKE_NP}
	-@if [ "${DSYMUTIL}" != "true" -a -f ${INSTALL_LIB_DIR}/libpetiga.${SL_LINKER_SUFFIX} ]; then \
        ${DSYMUTIL} ${INSTALL_LIB_DIR}/libpetiga.${SL_LINKER_SUFFIX}; fi
cmake-install:
	@cd ${PETIGA_DIR}/${PETIGA_ARCH} && ${OMAKE} install
cmake-clean:
	@if [ -f ${PETIGA_DIR}/${PETIGA_ARCH}/Makefile ]; then \
	cd ${PETIGA_DIR}/${PETIGA_ARCH} && ${OMAKE} clean; fi;
all-cmake: chk_petsc_dir chk_petiga_dir arch-tree
	-@echo "=================================================="
	-@echo "Building PetIGA (CMake - ${MAKE_NP} build jobs)"
	-@echo "Using PETIGA_DIR=${PETIGA_DIR}"
	-@echo "Using PETIGA_ARCH=${PETIGA_ARCH}"
	-@echo "Using PETSC_DIR=${PETSC_DIR}"
	-@echo "Using PETSC_ARCH=${PETSC_ARCH}"
	-@echo "=================================================="
	@${OMAKE} cmake-build PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR} PETIGA_ARCH=${PETIGA_ARCH} 2>&1 | tee ./${PETIGA_ARCH}/log/make.log
	-@echo "=================================================="
.PHONY: cmake-boot cmake-down cmake-build cmake-clean all-cmake


#
# Legacy build
#
legacy-build: arch-tree deletelibs deletemods build
legacy-clean: deletemods deletelibs
	-@${OMAKE} tree ACTION=clean PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR} PETIGA_ARCH=${PETIGA_ARCH}
all-legacy: chk_petsc_dir chk_petiga_dir arch-tree
	-@echo "=================================================="
	-@echo "Building PetIGA (legacy build)"
	-@echo "Using PETIGA_DIR=${PETIGA_DIR}"
	-@echo "Using PETIGA_ARCH=${PETIGA_ARCH}"
	-@echo "Using PETSC_DIR=${PETSC_DIR}"
	-@echo "Using PETSC_ARCH=${PETSC_ARCH}"
	-@echo "=================================================="
	-@echo "Beginning to build PetIGA library"
	@${OMAKE} legacy-build PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR} PETIGA_ARCH=${PETIGA_ARCH} 2>&1 | tee ./${PETSC_ARCH}/log/make.log
	-@echo "Completed building PetIGA library"
	-@echo "=================================================="
.PHONY: legacy-build legacy-clean all-legacy

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
build: compile ranlib shlibs
compile:
	-@${OMAKE} tree ACTION=libfast PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR} PETIGA_ARCH=${PETIGA_ARCH}
	-@${MV} -f ${PETIGA_DIR}/src/petiga*.mod ${PETIGA_DIR}/${PETIGA_ARCH}/include
ranlib:
	-@echo "building libpetiga.${AR_LIB_SUFFIX}"
	-@${RANLIB} ${PETIGA_LIB_DIR}/*.${AR_LIB_SUFFIX} > tmpf 2>&1 ; ${GREP} -v "has no symbols" tmpf; ${RM} tmpf;
shlibs:
	-@${OMAKE} shared_nomesg PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR} PETIGA_ARCH=${PETIGA_ARCH} \
		   | (${GREP} -vE "making shared libraries in" || true) \
		   | (${GREP} -vE "==========================" || true)
	-@if [ "${DSYMUTIL}" != "true" ]; then \
        ${DSYMUTIL} ${INSTALL_LIB_DIR}/libpetiga.${SL_LINKER_SUFFIX}; fi
.PHONY: build compile ranlib shlibs

# Delete PetIGA library
deletelogs:
	-@${RM} -r ${PETIGA_DIR}/${PETIGA_ARCH}/log/*.log
deletemods:
	-@${RM} -r ${PETIGA_DIR}/${PETIGA_ARCH}/include/petiga*.mod
deletestaticlibs:
	-@${RM} -r ${PETIGA_LIB_DIR}/libpetiga*.${AR_LIB_SUFFIX}
deletesharedlibs:
	-@${RM} -r ${PETIGA_LIB_DIR}/libpetiga*.${SL_LINKER_SUFFIX}*
deletelibs: deletestaticlibs deletesharedlibs
.PHONY: deletelogs deletemods deletestaticlibs deletesharedlibs deletelibs


# Clean up build
clean:: allclean
allclean:
	@if [ "${MAKE_IS_GNUMAKE}" != "" ]; then \
	${OMAKE} gmake-clean  PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR} PETIGA_ARCH=${PETIGA_ARCH}; \
	elif [ "${PETSC_BUILD_USING_CMAKE}" != "" ]; then \
	${OMAKE} cmake-clean  PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR} PETIGA_ARCH=${PETIGA_ARCH}; else \
	${OMAKE} legacy-clean PETSC_ARCH=${PETSC_ARCH} PETSC_DIR=${PETSC_DIR} PETIGA_DIR=${PETIGA_DIR} PETIGA_ARCH=${PETIGA_ARCH}; fi;
distclean: chk_petiga_dir
	@echo "*** Deleting all build files ***"
	-${RM} -r ${PETIGA_DIR}/${PETIGA_ARCH}/
.PHONY: clean allclean distclean


# Test build
check:
	-@echo "=================================================="
	-@echo "Running check to verify correct installation"
	-@echo "=================================================="
	-@echo "Using PETIGA_DIR=${PETIGA_DIR}"
	-@echo "Using PETIGA_ARCH=${PETIGA_ARCH}"
	-@echo "Using PETSC_DIR=${PETSC_DIR}"
	-@echo "Using PETSC_ARCH=${PETSC_ARCH}"
	@${OMAKE} -C test clean
	@${OMAKE} -C test check clean
	-@echo "Completed compiling and running check"
	-@echo "=================================================="
.PHONY: check


# Run tests
test:
	-@echo "=================================================="
	-@echo "Beginning to compile and run test examples"
	-@echo "=================================================="
	@${OMAKE} -C test test clean
	@${OMAKE} -C demo test clean
	-@echo "Completed compiling and running tests"
	-@echo "=================================================="
.PHONY: test


#
# Documentation
#
SRCDIR=${PETIGA_DIR}/src
DOCDIR=${PETIGA_DIR}/docs/html
doc:
	@if [ ! -d ${DOCDIR} ]; then ${MKDIR} ${DOCDIR}; fi
	-@${RM} ${DOCDIR}/*.html
	@${PETSC_DIR}/${PETSC_ARCH}/bin/doctext -mpath ${DOCDIR} -html ${SRCDIR}/*.c
	@echo '<TITLE>PetIGA Documentation</TITLE>' > ${DOCDIR}/index.html
	@echo '<H1>PetIGA Documentation</H1>' >> ${DOCDIR}/index.html
	@echo '<MENU>' >> ${DOCDIR}/index.html
	@ls -1 ${DOCDIR} | grep .html | grep -v index.html | sed -e 's%^\(.*\).html$$%<LI><A HREF="\1.html">\1</A>%g' >> ${DOCDIR}/index.html
	@echo '</MENU>' >> ${DOCDIR}/index.html
deletedoc:
	-@${RM} ${DOCDIR}/*.html
.PHONY: doc deletedoc


#
# TAGS Generation
#
alletags:
	-@${PYTHON} ${PETSC_DIR}/bin/maint/generateetags.py
deleteetags:
	-@${RM} CTAGS TAGS
.PHONY: alletags deleteetags
