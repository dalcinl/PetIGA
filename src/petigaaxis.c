#include "petiga.h"

#undef  __FUNCT__
#define __FUNCT__ "IGAAxisCreate"
PetscErrorCode IGAAxisCreate(IGAAxis *_axis)
{
  IGAAxis        axis;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(_axis,1);
  ierr = PetscCalloc1(1,_axis);CHKERRQ(ierr);
  (*_axis)->refct = 1; axis = *_axis;

  /* */
  axis->periodic = PETSC_FALSE;
  /* */
  ierr = PetscMalloc1(2,&axis->U);CHKERRQ(ierr);
  axis->p = 0;
  axis->m = 1;
  axis->U[0] = -0.5;
  axis->U[1] = +0.5;
  /* */
  ierr = PetscMalloc1(1,&axis->span);CHKERRQ(ierr);
  axis->nnp = 1;
  axis->nel = 1;
  axis->span[0] = 0;

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
  axis = *_axis; *_axis = NULL;
  if (!axis) PetscFunctionReturn(0);
  if (--axis->refct > 0) PetscFunctionReturn(0);
  ierr = PetscFree(axis->U);CHKERRQ(ierr);
  ierr = PetscFree(axis->span);CHKERRQ(ierr);
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

  axis->periodic = PETSC_FALSE;

  if (axis->m != 1) {
    ierr = PetscFree(axis->U);CHKERRQ(ierr);
    ierr = PetscMalloc1(2,&axis->U);CHKERRQ(ierr);
  }
  axis->p = 0;
  axis->m = 1;
  axis->U[0] = -0.5;
  axis->U[1] = +0.5;

  if (axis->nel != 1) {
    ierr = PetscFree(axis->span);CHKERRQ(ierr);
    ierr = PetscMalloc1(1,&axis->span);CHKERRQ(ierr);
  }
  axis->nnp = 1;
  axis->nel = 1;
  axis->span[0] = 0;

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
/*@
   IGAAxisCopy - Copies an axis. axis <-- base

   Logically Collective on IGAAxis

   Input Parameters:
.  base - the axis

   Input Parameters:
.  axis - the copy

   Level: normal

.keywords: IGA, axis, copy
@*/
PetscErrorCode IGAAxisCopy(IGAAxis base,IGAAxis axis)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(base,1);
  PetscValidPointer(axis,2);
  if (base == axis) PetscFunctionReturn(0);

  axis->periodic = base->periodic;

  axis->p = base->p;
  axis->m = base->m;
  ierr = PetscFree(axis->U);CHKERRQ(ierr);
  ierr = PetscMalloc1(axis->m+1,&axis->U);CHKERRQ(ierr);
  ierr = PetscMemcpy(axis->U,base->U,(axis->m+1)*sizeof(PetscReal));CHKERRQ(ierr);

  axis->nnp = base->nnp;
  axis->nel = base->nel;
  ierr = PetscFree(axis->span);CHKERRQ(ierr);
  ierr = PetscMalloc1(axis->nel,&axis->span);CHKERRQ(ierr);
  ierr = PetscMemcpy(axis->span,base->span,axis->nel*sizeof(PetscInt));CHKERRQ(ierr);

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
  ierr = PetscCalloc1(1,axis);CHKERRQ(ierr);
  (*axis)->refct = 1;
  ierr = IGAAxisCopy(base,*axis);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAAxisSetPeriodic"
/*@
   IGAAxisSetPeriodic - Sets the axis periodicity

   Logically Collective on IGAAxis

   Input Parameters:
+  axis - the IGAAxis context
-  periodic - TRUE or FALSE

   Level: normal

.keywords: IGA, axis, periodic
@*/
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
#define __FUNCT__ "IGAAxisSetDegree"
/*@
   IGAAxisSetDegree - Sets the axis degree

   Logically Collective on IGAAxis

   Input Parameters:
+  axis - the IGAAxis context
-  p - the polynomial degree

   Level: normal

.keywords: IGA, axis, degree
@*/
PetscErrorCode IGAAxisSetDegree(IGAAxis axis,PetscInt p)
{
  PetscFunctionBegin;
  PetscValidPointer(axis,1);
  if (p < 1)
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
             "Polynomial degree must be greater than one, got %D",p);
  if (axis->p > 0 && axis->m > 1 && axis->p != p)
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,
             "Cannot change degree to %D after it was set to %D",p,axis->p);
  axis->p = p;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAAxisGetDegree"
PetscErrorCode IGAAxisGetDegree(IGAAxis axis,PetscInt *p)
{
  PetscFunctionBegin;
  PetscValidPointer(axis,1);
  PetscValidPointer(p,2);
  *p = axis->p;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAAxisSetKnots"
PetscErrorCode IGAAxisSetKnots(IGAAxis axis,PetscInt m,const PetscReal U[])
{
  PetscInt       p,n,k;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(axis,1);
  PetscValidPointer(U,3);

  if (axis->p < 1)
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,
            "Must call IGAAxisSetDegree() first");
  if (m < 2*axis->p+1)
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
             "Number of knots must be at least %D, got %D",
             2*(axis->p+1),m+1);

  p = axis->p;
  n = m -p - 1;

  for (k=1; k<=m;) {
    PetscInt i = k, s = 1;
    /* check increasing sequence */
    if (U[k-1] > U[k])
      SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
               "Knot sequence must be increasing, got U[%D]=%g > U[%D]=%g",
               k-1,(double)U[k-1],k,(double)U[k]);
    /* check multiplicity */
    while (++k < m && U[i] == U[k]) s++;
    if (s > p)
      SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
               "Knot U[%D]=%g has multiplicity %D greater than polynomial degree %D",
               i,(double)U[i],s,p);
  }

  if (m != axis->m) {
    PetscReal *V;
    ierr = PetscMalloc1(m+1,&V);CHKERRQ(ierr);
    ierr = PetscFree(axis->U);CHKERRQ(ierr);
    axis->m = m;
    axis->U = V;
  }
  ierr = PetscMemcpy(axis->U,U,(m+1)*sizeof(PetscReal));CHKERRQ(ierr);

  axis->nel = 0;
  ierr = PetscFree(axis->span);CHKERRQ(ierr);
  ierr = IGAAxisGetSpans(axis,&axis->nel,&axis->span);CHKERRQ(ierr);

  if (axis->periodic) {
    PetscInt s = 1;
    while (s < p && U[m-p] == U[m-p+s]) s++;
    axis->nnp = n-p+s;
  } else {
    axis->nnp = n+1;
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
#define __FUNCT__ "IGAAxisGetLimits"
PetscErrorCode IGAAxisGetLimits(IGAAxis axis,PetscReal *Ui,PetscReal *Uf)
{
  PetscFunctionBegin;
  PetscValidPointer(axis,1);
  if (Ui) PetscValidRealPointer(Ui,2);
  if (Uf) PetscValidRealPointer(Uf,3);
  if (Ui) *Ui = axis->U[axis->p];
  if (Uf) *Uf = axis->U[axis->m-axis->p];
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAAxisGetSizes"
PetscErrorCode IGAAxisGetSizes(IGAAxis axis,PetscInt *nel,PetscInt *nnp)
{
  PetscFunctionBegin;
  PetscValidPointer(axis,1);
  if (nel) PetscValidIntPointer(nel,2);
  if (nnp) PetscValidIntPointer(nnp,3);
  if (nel) *nel = axis->nel;
  if (nnp) *nnp = axis->nnp;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
extern PetscInt IGA_SpanCount(PetscInt n,PetscInt p,const PetscReal U[]);
extern PetscInt IGA_SpanIndex(PetscInt n,PetscInt p,const PetscReal U[],PetscInt index[]);
EXTERN_C_END

#undef  __FUNCT__
#define __FUNCT__ "IGAAxisGetSpans"
PetscErrorCode IGAAxisGetSpans(IGAAxis axis,PetscInt *nel,PetscInt *span[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(axis,1);
  if (nel)  PetscValidIntPointer(nel,2);
  if (span) PetscValidPointer(span,3);
  if (!axis->span) {
    PetscInt p = axis->p;
    PetscInt m = axis->m;
    PetscInt n = m - p - 1;
    axis->nel = IGA_SpanCount(n,p,axis->U);
    ierr = PetscMalloc1(axis->nel,&axis->span);CHKERRQ(ierr);
    (void)IGA_SpanIndex(n,p,axis->U,axis->span);
  }
  if (nel)  *nel  = axis->nel;
  if (span) *span = axis->span;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAAxisInit"
PetscErrorCode IGAAxisInit(IGAAxis axis,PetscInt p,PetscInt m,const PetscReal U[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(axis,1);
  axis->p = 0;
  ierr = IGAAxisSetDegree(axis,p);CHKERRQ(ierr);
  ierr = IGAAxisSetKnots(axis,m,U);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAAxisInitBreaks"
PetscErrorCode IGAAxisInitBreaks(IGAAxis axis,PetscInt nu,const PetscReal u[],PetscInt C)
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
            "Must call IGAAxisSetDegree() first");
  if (nu < 2)
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
             "Number of breaks must be at least two, got %D",nu);
  for (i=1; i<nu; i++)
    if (u[i-1] >= u[i])
      SETERRQ5(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
               "Break sequence must be strictly increasing, "
               "got u[%D]=%g %s u[%D]=%g",
               i-1,(double)u[i-1],(u[i]==u[i-1])?"==":">",i,(double)u[i]);
  if (C < 0 || C >= axis->p)
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,
             "Continuity must be in range [0,%D], got %D",axis->p-1,C);

  p = axis->p; /* polynomial degree */
  s = p - C; /* multiplicity */
  r = nu - 1; /* last break index */
  m = 2*(p+1) + (r-1)*s - 1; /* last knot index */
  n = m - p - 1; /* last basis function index */

  if (m != axis->m) {
    ierr = PetscMalloc1(m+1,&U);CHKERRQ(ierr);
    ierr = PetscFree(axis->U);CHKERRQ(ierr);
    axis->m = m;
    axis->U = U;
  } else {
    U = axis->U;
  }

  for (k=0; k<=p; k++) { /* open part */
    U[k]   = u[0];
    U[m-k] = u[r];
  }
  for (i=1; i<=r-1; i++) { /* r-1 breaks */
    for (j=0; j<s; j++)    /* s times */
      U[k++] = u[i];
  }
  if (axis->periodic) {
    for (k=0; k<=C; k++) { /* periodic part */
      U[C-k]   = U[p] - U[m-p] + U[n-k];
      U[m-C+k] = U[m-p] - U[p] + U[p+1+k];
    }
  }

  axis->nel = r;
  ierr = PetscFree(axis->span);CHKERRQ(ierr);
  ierr = PetscMalloc1(axis->nel,&axis->span);CHKERRQ(ierr);
  for (i=0; i<axis->nel; i++) axis->span[i] = p + i*s;

  axis->nnp = axis->periodic ? n-C : n+1;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAAxisInitUniform"
/*@
   IGAAxisInitUniform - Initializes an axis with uniformly spaces knots.

   Logically Collective on IGAAxis

   Input Parameters:
+  axis - the IGAAxis context
.  N - the number of equally-spaced, nonzero spans (elements)
.  Ui - the initial knot value
.  Uf - the final knot value
-  C - the global continuity order

   Notes: You must have called IGAAxisSetDegree() prior to this
   command. Creates a function space which consists of N spans of
   piecewise polynomials of the degree set with continuity order C at
   element interfaces.

   Level: normal

.keywords: IGA, axis, initialize, uniform
@*/
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
            "Must call IGAAxisSetDegree() first");
  if (N < 1)
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,
             "Number of elements must be greater than zero, got %D",N);
  if (Ui >= Uf)
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,
             "Initial value %g must be less than final value %g",(double)Ui,(double)Uf);
  if (C < 0 || C >= axis->p)
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,
             "Continuity must be in range [0,%D], got %D",axis->p-1,C);

  p = axis->p; /* polynomial degree */
  s = p - C; /* multiplicity */
  r = N ; /* last break index */
  m = 2*(p+1) + (N-1)*s - 1; /* last knot index */
  n = m - p - 1; /* last basis function index */

  if (m != axis->m) {
    ierr = PetscMalloc1(m+1,&U);CHKERRQ(ierr);
    ierr = PetscFree(axis->U);CHKERRQ(ierr);
    axis->m = m;
    axis->U = U;
  } else {
    U = axis->U;
  }

  for (k=0; k<=p; k++) { /* open part */
    U[k]   = Ui;
    U[m-k] = Uf;
  }
  for (i=1; i<=r-1; i++) { /* (N-1) breaks */
    for (j=1; j<=s; j++)     /* s times */
      U[k++] = Ui + i * ((Uf-Ui)/N);
  }
  if (axis->periodic) {
    for (k=0; k<=C; k++) { /* periodic part */
      U[C-k]   = U[p] - U[m-p] + U[n-k];
      U[m-C+k] = U[m-p] - U[p] + U[p+1+k];
    }
  }

  axis->nel = r;
  ierr = PetscFree(axis->span);CHKERRQ(ierr);
  ierr = PetscMalloc1(axis->nel,&axis->span);CHKERRQ(ierr);
  for (i=0; i<axis->nel; i++) axis->span[i] = p + i*s;

  axis->nnp = axis->periodic ? n-C : n+1;

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAAxisSetUp"
PetscErrorCode IGAAxisSetUp(IGAAxis axis)
{
  PetscInt        p,m,n;
  const PetscReal *U;
  PetscErrorCode  ierr;
  PetscFunctionBegin;
  PetscValidPointer(axis,1);
  if (axis->p < 1)
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,
            "Must call IGAAxisSetDegree() first");
  if (axis->m < 2*axis->p+1)
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,
            "Must call IGAAxisSetKnots() first");

  p = axis->p;
  m = axis->m;
  n = m - p - 1;
  U = axis->U;

  ierr = IGAAxisGetSpans(axis,&axis->nel,&axis->span);CHKERRQ(ierr);

  if (axis->periodic) {
    PetscInt s = 1;
    while (s < p && U[m-p] == U[m-p+s]) s++;
    axis->nnp = n-p+s;
  } else {
    axis->nnp = n+1;
  }
  PetscFunctionReturn(0);
}

PetscInt IGA_SpanCount(PetscInt n,PetscInt p,const PetscReal U[])
{
  PetscInt i, span = 0;
  for (i=p; i<=n; i++)
    if (U[i] != U[i+1])
      span++;
  return span;
}

PetscInt IGA_SpanIndex(PetscInt n,PetscInt p,const PetscReal U[],PetscInt index[])
{
  PetscInt i, span = 0;
  for (i=p; i<=n; i++)
    if (U[i] != U[i+1])
      index[span++] = i;
  return span;
}

/*
void IGA_Stencil(PetscInt n,PetscInt p,const PetscReal U[],
                 PetscBool periodic,PetscInt first[],PetscInt last[])
{
  PetscInt i,k;
  for (i=0; i<=n; i++) {
    k = i;
    while (U[k]==U[k+1]) k++;
    first[i] = k - p;
    k = i + p + 1;
    while (U[k]==U[k-1]) k--;
    last[i] = k-1;
  }
  if (!periodic)
    for (i=0; i<=p; i++) {
      first[i]  = 0;
      last[n-i] = n;
    }
  else {
    PetscInt s = 1;
    while (s < p && U[m-p]==U[m-p+s]) s++;
    k = n - p + s;
    while (U[k]==U[k+1]) k++;
    first[0] = k - s - n;
  }
}
*/
