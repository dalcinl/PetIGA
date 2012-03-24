#include "petiga.h"

#undef  __FUNCT__
#define __FUNCT__ "IGAAxisCreate"
PetscErrorCode IGAAxisCreate(IGAAxis *_axis)
{
  IGAAxis        axis;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(_axis,1);
  ierr = PetscNew(struct _n_IGAAxis,_axis);CHKERRQ(ierr);
  (*_axis)->refct = 1; axis = *_axis;
  axis->periodic = PETSC_FALSE;
  axis->p = 0;
  axis->m = 1;
  ierr = PetscMalloc((axis->m+1)*sizeof(PetscReal),&axis->U);CHKERRQ(ierr);
  axis->U[0] = -0.5;
  axis->U[1] = +0.5;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAAxisDestroy"
PetscErrorCode IGAAxisDestroy(IGAAxis *_axis)
{
  IGAAxis        axis;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(_axis,1);
  axis = *_axis; *_axis = 0;
  if (!axis) PetscFunctionReturn(0);
  if (--axis->refct > 0) PetscFunctionReturn(0);
  ierr = PetscFree(axis->U);CHKERRQ(ierr);
  ierr = PetscFree(axis);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAAxisReset"
PetscErrorCode IGAAxisReset(IGAAxis axis)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!axis) PetscFunctionReturn(0);
  PetscValidPointer(axis,1);
  if (axis->p == 0 && axis->m == 1) PetscFunctionReturn(0);
  ierr = PetscFree(axis->U);CHKERRQ(ierr);
  axis->periodic = PETSC_FALSE;
  axis->p = 0;
  axis->m = 1;
  ierr = PetscMalloc((axis->m+1)*sizeof(PetscReal),&axis->U);CHKERRQ(ierr);
  axis->U[0] = -0.5;
  axis->U[1] = +0.5;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAAxisReference"
PetscErrorCode IGAAxisReference(IGAAxis axis)
{
  PetscFunctionBegin;
  PetscValidPointer(axis,1);
  axis->refct++;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAAxisCopy"
PetscErrorCode IGAAxisCopy(IGAAxis base,IGAAxis axis)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(base,1);
  PetscValidPointer(axis,2);
  axis->periodic = base->periodic;
  axis->p = base->p;
  axis->m = base->m;
  ierr = PetscFree(axis->U);CHKERRQ(ierr);
  ierr = PetscMalloc((axis->m+1)*sizeof(PetscReal),&axis->U);CHKERRQ(ierr);
  ierr = PetscMemcpy(axis->U,base->U,(axis->m+1)*sizeof(PetscReal));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAAxisDuplicate"
PetscErrorCode IGAAxisDuplicate(IGAAxis base,IGAAxis *axis)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(base,1);
  PetscValidPointer(axis,2);
  ierr = PetscNew(struct _n_IGAAxis,axis);CHKERRQ(ierr);
  ierr = IGAAxisCopy(base,*axis);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAAxisSetPeriodic"
PetscErrorCode IGAAxisSetPeriodic(IGAAxis axis,PetscBool periodic)
{
  PetscFunctionBegin;
  PetscValidPointer(axis,1);
  axis->periodic = periodic ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAAxisGetPeriodic"
PetscErrorCode IGAAxisGetPeriodic(IGAAxis axis,PetscBool *periodic)
{
  PetscFunctionBegin;
  PetscValidPointer(axis,1);
  PetscValidPointer(periodic,2);
  *periodic = axis->periodic;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAAxisSetOrder"
PetscErrorCode IGAAxisSetOrder(IGAAxis axis,PetscInt p)
{
  PetscFunctionBegin;
  PetscValidPointer(axis,1);
  if (p < 1)
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
             "Polynomial order must be greather than zero, got %D",p);
  axis->p = p;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAAxisGetOrder"
PetscErrorCode IGAAxisGetOrder(IGAAxis axis,PetscInt *p)
{
  PetscFunctionBegin;
  PetscValidPointer(axis,1);
  PetscValidPointer(p,2);
  *p = axis->p;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAAxisSetKnots"
PetscErrorCode IGAAxisSetKnots(IGAAxis axis,PetscInt m,PetscReal U[])
{
  PetscReal      *V = 0;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(axis,1);
  if (U) PetscValidPointer(U,3);
  if (m < 2*axis->p+1)
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,
             "Number of knots must be at least %D, got %D",2*(axis->p+1),m+1);
  if (U) {
    ierr = PetscMalloc((m+1)*sizeof(PetscReal),&V);CHKERRQ(ierr);
    ierr = PetscMemcpy(V,U,(m+1)*sizeof(PetscReal));CHKERRQ(ierr);
    ierr = PetscFree(axis->U);CHKERRQ(ierr);
    axis->m = m;
    axis->U = V;
  } else if(m != axis->m || !axis->U) {
    ierr = PetscMalloc((m+1)*sizeof(PetscReal),&V);CHKERRQ(ierr);
    ierr = PetscMemzero(V,(m+1)*sizeof(PetscReal));CHKERRQ(ierr);
    ierr = PetscFree(axis->U);CHKERRQ(ierr);
    axis->m = m;
    axis->U = V;
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAAxisGetKnots"
PetscErrorCode IGAAxisGetKnots(IGAAxis axis,PetscInt *m,PetscReal *U[])
{
  PetscFunctionBegin;
  PetscValidPointer(axis,1);
  if (m) PetscValidPointer(m,2);
  if (U) PetscValidPointer(U,3);
  if (m) *m = axis->m;
  if (U) *U = axis->U;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAAxisInitUniform"
PetscErrorCode IGAAxisInitUniform(IGAAxis axis,PetscInt p,PetscInt C,
                                  PetscInt E,PetscReal Ui,PetscReal Uf)
{
  PetscInt       m,k,i,j;
  PetscReal      *U;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(axis,1);

  if (p == PETSC_DEFAULT) p = axis->p;
  if (C == PETSC_DECIDE)  C = p-1;
  if (C == PETSC_DEFAULT) C = p-1;

  if (E < 1)
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,
             "Number of elements must be grather than zero, got %D",E);
  if (p < 1)
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,
             "Polynomial order must be grather than zero, got %D",p);
  if (C < 0 || C >= p)
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,
             "Continuity must be in range [0,%D), got %D",p,C);

  m = 2*(p+1) + (E-1)*(p-C) - 1; /* last knot index */
  ierr = PetscMalloc((m+1)*sizeof(PetscReal),&U);CHKERRQ(ierr);

  for(k=0; k<=p; k++) { /* open part */
    U[k]   = Ui;
    U[m-k] = Uf;
  }
  for(i=1; i<=(E-1); i++) { /* (E-1) knots */
    for(j=1; j<=(p-C); j++)   /* (p-C) times */
      U[k++] = Ui + i * ((Uf-Ui)/E);
  }
  if (axis->periodic) {
    for(k=0; k<=C; k++) { /* periodic part */
      U[k]     = Ui - Uf + U[m-C-(p+1)+k];
      U[m-C+k] = Uf - Ui + U[(p+1)+k];
    }
  }

  ierr = PetscFree(axis->U);CHKERRQ(ierr);
  axis->p = p;
  axis->m = m;
  axis->U = U;

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAAxisCheck"
PetscErrorCode IGAAxisCheck(IGAAxis axis)
{
  PetscFunctionBegin;
  PetscValidPointer(axis,1);
  if (axis->p < 1)
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,
            "Must call IGAAxisSetOrder() first");
  if (!axis->U || axis->m < 2*axis->p+1)
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,
            "Must call IGAAxisSetKnots() first");
  PetscFunctionReturn(0);
}
