#requiresdefine   'PETSC_HAVE_FORTRAN'
#requiresprecision double
#requiresscalar    real

ALL: lib

#CFLAGS = -g3 -Wall -Wextra -Wno-unused-parameter -Wconversion -Wno-sign-conversion
#FFLAGS = -g3 -Wall -Wextra -fcheck=all

SOURCEH  = petiga.h
SOURCEC  = petiga.c petigareg.c petigaaxis.c petigarule.c petigabasis.c petigaelem.c petigapoint.c petigaksp.c petigasnes.c petigats.c petigabound.c petigaio.c petigapc.c
SOURCEF  = petigabsp.f90 petiga1d.f90 petiga2d.f90 petiga3d.f90 petigaint.f90
OBJSC    = ${SOURCEC:.c=.o}
OBJSF    = ${SOURCEF:.f90=.o}

LIBBASE  = libpetiga
DIRS     =
LOCDIR   = src/
MANSEC   = IGA

include ${PETIGA_DIR}/conf/petigavariables
include ${PETIGA_DIR}/conf/petigarules
include ${PETIGA_DIR}/conf/petigatest
