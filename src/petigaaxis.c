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
  ierr = PetscMalloc(2*sizeof(PetscReal),&axis->U);CHKERRQ(ierr);
  axis->periodic = PETSC_FALSE;
  axis->p = 0;
  axis->m = 1;
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
  if (axis->m != 1) {
    ierr = PetscFree(axis->U);CHKERRQ(ierr);
    ierr = PetscMalloc(2*sizeof(PetscReal),&axis->U);CHKERRQ(ierr);
  }
  axis->periodic = PETSC_FALSE;
  axis->p = 0;
  axis->m = 1;
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
  PetscInt       k;
  PetscReal      *V = 0;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(axis,1);
  if (U) PetscValidPointer(U,3);
  /*
  if (axis->p < 1)
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,
            "Must call IGAAxisSetOrder() first");
  */
  if (m < 2*axis->p+1)
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
             "Number of knots must be at least %D, got %D",2*(axis->p+1),m+1);
  if (U) for (k=1; k<=m; k++)
           if (U[k-1] > U[k])
             SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
                      "Knot sequence must be increasing, "
                      "got U[%D]=%G > U[%D]=%G",
                      k-1,U[k-1],k,U[k]);

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
#define __FUNCT__ "IGAAxisInitBreaks"
PetscErrorCode IGAAxisInitBreaks(IGAAxis axis,PetscInt nu,PetscReal u[],PetscInt C)
{
  PetscInt       i,j,k;
  PetscInt       p,s,n,m,r;
  PetscReal      *U;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(axis,1);
  PetscValidPointer(u,3);

  if (C == PETSC_DECIDE) C = axis->p-1;

  if (axis->p < 1)
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,
            "Must call IGAAxisSetOrder() first");
  if (nu < 2)
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
             "Number of breaks must be at least two, got %D",nu);
  for (i=1; i<nu; i++)
    if (u[i-1] >= u[i])
      SETERRQ5(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
               "Break sequence must be strictly increasing, "
               "got u[%D]=%G %s u[%D]=%G",
               i-1,u[i-1],i,u[i],(u[i]==u[i-1])?"==":">");
  if (C < 0 || C >= axis->p)
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,
             "Continuity must be in range [0,%D], got %D",axis->p-1,C);

  p = axis->p; /* polynomial order */
  s = p - C; /* multiplicity */
  r = nu - 1; /* last break index */
  m = 2*(p+1) + (r-1)*s - 1; /* last knot index */
  n = m - p - 1; /* last basis function index */
  ierr = PetscMalloc1(m+1,PetscReal,&U);CHKERRQ(ierr);

  for(k=0; k<=p; k++) { /* open part */
    U[k]   = u[0];
    U[m-k] = u[r];
  }
  for(i=1; i<=r-1; i++) { /* r-1 breaks */
    for(j=0; j<s; j++)    /* s times */
      U[k++] = u[i];
  }
  if (axis->periodic) {
    for(k=0; k<=C; k++) { /* periodic part */
      U[k]     = U[p] - U[m-p] + U[n-C+k];
      U[m-C+k] = U[m-p] - U[p] + U[p+1+k];
    }
  }

  ierr = PetscFree(axis->U);CHKERRQ(ierr);
  axis->p = p;
  axis->m = m;
  axis->U = U;

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAAxisInitUniform"
PetscErrorCode IGAAxisInitUniform(IGAAxis axis,PetscInt N,PetscReal Ui,PetscReal Uf,PetscInt C)
{
  PetscInt       i,j,k;
  PetscInt       p,s,n,m,r;
  PetscReal      *U;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(axis,1);

  if (C == PETSC_DECIDE) C = axis->p-1;

  if (axis->p < 1)
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,
            "Must call IGAAxisSetOrder() first");
  if (N < 1)
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,
             "Number of elements must be grather than zero, got %D",N);
  if (Ui >= Uf)
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,
             "Initial value %G must be less than final value %G",Ui,Uf);
  if (C < 0 || C >= axis->p)
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,
             "Continuity must be in range [0,%D], got %D",axis->p-1,C);

  p = axis->p;  /* polynomial order */
  s = p - C; /* multiplicity */
  r = N ; /* last break index */
  m = 2*(p+1) + (N-1)*s - 1; /* last knot index */
  n = m - p - 1; /* last basis function index */
  ierr = PetscMalloc1(m+1,PetscReal,&U);CHKERRQ(ierr);

  for(k=0; k<=p; k++) { /* open part */
    U[k]   = Ui;
    U[m-k] = Uf;
  }
  for(i=1; i<=r-1; i++) { /* (N-1) breaks */
    for(j=1; j<=s; j++)     /* s times */
      U[k++] = Ui + i * ((Uf-Ui)/N);
  }
  if (axis->periodic) {
    for(k=0; k<=C; k++) { /* periodic part */
      U[k]     = U[p] - U[m-p] + U[n-C+k];
      U[m-C+k] = U[m-p] - U[p] + U[p+1+k];
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
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,
            "Must call IGAAxisSetOrder() first");
  if (!axis->U || axis->m < 2*axis->p+1)
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,
            "Must call IGAAxisSetKnots() first");
#ifdef PETSC_USE_DEBUG
  {
    PetscInt  k  = 1;
    PetscInt  p  = axis->p;
    PetscInt  m  = axis->m;
    PetscReal *U = axis->U;
    while (k <= m) {
      PetscInt i=k,s=1;
      if (U[k-1] > U[k])
        SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
                 "Knot sequence must be increasing, "
                 "got U[%D]=%G > U[%D]=%G",
                 k-1,U[k-1],k,U[k]);
      while (++k < m && U[k-1] == U[k]) s++;
      if (s > p)
        SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
                 "Knot U[%D]=%G has multiplicity %D "
                 "greather than polynomial order %D",
                 i,U[i],s,p);
    }
  }
#endif
  PetscFunctionReturn(0);
}
