#requiresdefine   'PETSC_HAVE_FORTRAN'

ALL: lib

#CFLAGS = -g3 -Wall -Wextra -Wno-unused-parameter -Wconversion -Wno-sign-conversion
#FFLAGS = -g3 -Wall -Wextra -fcheck=all

SOURCEH  = ../include/petiga.h petigapc.h petigagrid.h petigapart.h
SOURCEC  = petiga.c petigareg.c petigaaxis.c petigarule.c petigabasis.c petigabound.c petigaelem.c petigapoint.c petigavec.c petigamat.c petigascl.c petigapcb.c petigapce.c petigaksp.c petigasnes.c petigats.c petigaio.c petigagrid.c petigapart.c
SOURCEF1 = petigaftn.F90
SOURCEF2 = petigabsp.f90 petiga1d.f90 petiga2d.f90 petiga3d.f90 petigaint.f90
SOURCEF  = ${SOURCEF1} ${SOURCEF2}
OBJSC    = ${SOURCEC:.c=.o}
OBJSF    = ${SOURCEF1:.F90=.o} ${SOURCEF2:.f90=.o}

LIBBASE  = libpetiga
DIRS     =
LOCDIR   = src/
MANSEC   = IGA

include ${PETIGA_DIR}/conf/petigavariables
include ${PETIGA_DIR}/conf/petigarules
include ${PETIGA_DIR}/conf/petigatest
