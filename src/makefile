#requiresdefine   'PETSC_HAVE_FORTRAN'

ALL: lib

#CFLAGS_STD_C89 = -std=c89 -Dinline=__inline -Wno-long-long
#CFLAGS_STD_C99 = -std=c99
#CFLAGS_STD_C11 = -std=c11
#CFLAGS_STD = ${CFLAGS_STD_C89}
#CFLAGS = ${CFLAGS_STD} -pedantic -Wall -Wextra -Wno-unused-parameter -Wconversion -Wno-sign-conversion

#FFLAGS_STD_F03 = -std=f2003
#FFLAGS_STD_F08 = -std=f2008
#FFLAGS_STD_F08 = -std=f2008ts
#FFLAGS_STD = ${FFLAGS_STD_F03}
#FFLAGS = ${FFLAGS_STD} -pedantic -Wall -Wextra -Wimplicit-interface -g3 -fcheck=all -fbacktrace

SOURCEH = \
../include/petiga.h \
petigabl.h \
petigagrid.h \
petigapart.h \

SOURCEC = \
petiga.c \
petigareg.c \
petigapart.c \
petigagrid.c \
petigaio.c \
petigaaxis.c \
petigarule.c \
petigabasis.c \
petigaform.c \
petigaelem.c \
petigapoint.c \
petigavec.c \
petigamat.c \
petigadm.c \
petigacomp.c \
petigapcb.c \
petigapce.c \
petigapc.c \
petigaksp.c \
petigasnes.c \
petigats.c \
petigats2.c \
tsalpha2.c \
petscvwopt.c \
snesfdcolor.c \

SOURCEF = \
petigaftn.F90 \
petigaval.F90 \
petigabsp.F90 \
petigaqdr.F90 \
petiga1d.F90 \
petiga2d.F90 \
petiga3d.F90 \

OBJSC = ${SOURCEC:.c=.o}
OBJSF = ${SOURCEF:.F90=.o}

LIBBASE  = libpetiga
DIRS     =
LOCDIR   = src/
MANSEC   = IGA

include ${PETIGA_DIR}/conf/petigavariables
include ${PETIGA_DIR}/conf/petigarules
include ${PETIGA_DIR}/conf/petigatest
