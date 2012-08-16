#include "petiga.h"

EXTERN_C_BEGIN

PetscErrorCode igapointgetindex (IGAPoint p,PetscInt *i) {return IGAPointGetIndex(p,i);}
PetscErrorCode igapointgetcount (IGAPoint p,PetscInt *c) {return IGAPointGetCount(p,c);}

PetscErrorCode igapointformpoint     (IGAPoint p,PetscReal x[])               {return IGAPointFormPoint     (p,x);  }
PetscErrorCode igapointformgradmap   (IGAPoint p,PetscReal m[],PetscReal i[]) {return IGAPointFormGradMap   (p,m,i);}
PetscErrorCode igapointformshapefuns (IGAPoint p,PetscInt d,PetscReal N[])    {return IGAPointFormShapeFuns (p,d,N);}

PetscErrorCode igapointformvalue (IGAPoint p,const PetscScalar U[],PetscScalar u[]) {return IGAPointFormValue(p,U,u);}
PetscErrorCode igapointformgrad  (IGAPoint p,const PetscScalar U[],PetscScalar u[]) {return IGAPointFormGrad (p,U,u);}
PetscErrorCode igapointformhess  (IGAPoint p,const PetscScalar U[],PetscScalar u[]) {return IGAPointFormHess (p,U,u);}
PetscErrorCode igapointformdel2  (IGAPoint p,const PetscScalar U[],PetscScalar u[]) {return IGAPointFormDel2 (p,U,u);}
PetscErrorCode igapointformder3  (IGAPoint p,const PetscScalar U[],PetscScalar u[]) {return IGAPointFormDer3 (p,U,u);}

EXTERN_C_END
