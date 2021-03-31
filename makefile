# -*- mode: makefile-gmake -*-

ALL: all

PETIGA_DIR ?= $(CURDIR)
include ./lib/petiga/conf/variables

all:
	${OMAKE} gmake-all
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
gmake-build:
	@cd ${PETIGA_DIR} && ${OMAKE} -f gmakefile -j ${MAKE_NP}
gmake-clean:
	@cd ${PETIGA_DIR} && ${OMAKE} -f gmakefile clean
gmake-all: arch-tree
	-@echo "=================================================="
	-@echo "Building PetIGA (GNU Make - ${MAKE_NP} build jobs)"
	-@echo "Using PETIGA_DIR=${PETIGA_DIR}"
	-@echo "Using PETIGA_ARCH=${PETIGA_ARCH}"
	-@echo "Using PETSC_DIR=${PETSC_DIR}"
	-@echo "Using PETSC_ARCH=${PETSC_ARCH}"
	-@echo "=================================================="
	@${OMAKE} gmake-build 2>&1 | tee ./${PETIGA_ARCH}/log/make.log
	-@echo "=================================================="
.PHONY: gmake-build gmake-clean gmake-all


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
	@${OMAKE} gmake-clean
distclean:
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
# Install
#
PREFIX ?= /tmp/petiga
find-install-dir = \
find $2 -type d -exec \
install -m $1 -d "$(DESDIR)$(PREFIX)/{}" \;
find-install = \
find $2 -type f -name $3 -exec \
install -m $1 "{}" "$(DESDIR)$(PREFIX)/$(if $4,$4,{})" \;
install : all
	@echo "*** Installing PetIGA in PREFIX=$(PREFIX) ***"
	@$(call find-install-dir,755,include)
	@$(call find-install-dir,755,lib)
	@$(call find-install,644,include,'*.h')
	@$(call find-install,644,include,'*.hpp')
	@$(call find-install,644,lib/petiga/conf,'*')
	@$(call find-install,644,$(PETIGA_ARCH)/include,'petiga.mod',include)
	@$(call find-install,644,$(PETIGA_ARCH)/lib,'libpetiga*.$(AR_LIB_SUFFIX)',lib)
	@$(call find-install,755,$(PETIGA_ARCH)/lib,'libpetiga*.$(SL_LINKER_SUFFIX)',lib)
	@printf "override PETIGA_ARCH=\n" > $(DESDIR)$(PREFIX)/lib/petiga/conf/arch
	@printf "override PETSC_DIR=$(PETSC_DIR)\n" > $(DESDIR)$(PREFIX)/lib/petiga/conf/petsc
	@printf "override PETSC_ARCH=$(PETSC_ARCH)\n" >> $(DESDIR)$(PREFIX)/lib/petiga/conf/petsc
	@$(RM) $(PREFIX)/lib/petiga/conf/.DIR
.PHONY: install


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
	-@${PYTHON} ${PETSC_DIR}/lib/petsc/bin/maint/generateetags.py
deleteetags:
	-@${RM} CTAGS TAGS
.PHONY: alletags deleteetags
