#include "petiga.h"

const char *const IGABasisTypes[] = {
  "BSPLINE",
  "BERNSTEIN",
  "LAGRANGE",
  "SPECTRAL",
  /* */
  "IGABasisType","IGA_BASIS_",NULL};

PetscErrorCode IGABasisCreate(IGABasis *_basis)
{
  IGABasis       basis;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(_basis,3);
  ierr = PetscCalloc1(1,&basis);CHKERRQ(ierr);
  *_basis = basis; basis->refct = 1;
  PetscFunctionReturn(0);
}

PetscErrorCode IGABasisDestroy(IGABasis *_basis)
{
  IGABasis       basis;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(_basis,1);
  basis = *_basis; *_basis = NULL;
  if (!basis) PetscFunctionReturn(0);
  if (--basis->refct > 0) PetscFunctionReturn(0);
  ierr = IGABasisReset(basis);CHKERRQ(ierr);
  ierr = PetscFree(basis);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGABasisReset(IGABasis basis)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!basis) PetscFunctionReturn(0);
  PetscValidPointer(basis,1);
  basis->nel = 0;
  basis->nqp = 0;
  basis->nen = 0;
  ierr = PetscFree(basis->offset);CHKERRQ(ierr);
  ierr = PetscFree(basis->detJac);CHKERRQ(ierr);
  ierr = PetscFree(basis->weight);CHKERRQ(ierr);
  ierr = PetscFree(basis->point);CHKERRQ(ierr);
  ierr = PetscFree(basis->value);CHKERRQ(ierr);
  ierr = PetscFree(basis->bnd_value[0]);CHKERRQ(ierr);
  ierr = PetscFree(basis->bnd_value[1]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGABasisReference(IGABasis basis)
{
  PetscFunctionBegin;
  PetscValidPointer(basis,1);
  basis->refct++;
  PetscFunctionReturn(0);
}

PetscErrorCode IGABasisSetType(IGABasis basis,IGABasisType type)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(basis,1);
  if (basis->type != type) {ierr = IGABasisReset(basis);CHKERRQ(ierr);}
  basis->type = type;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
extern PetscInt IGA_NextKnot(PetscInt m,const PetscReal U[],PetscInt k,PetscInt direction);
EXTERN_C_END

EXTERN_C_BEGIN
extern void IGA_Basis_BSpline (PetscInt i,PetscReal u,PetscInt p,PetscInt d,const PetscReal U[],PetscReal B[]);
extern void IGA_Basis_Lagrange(PetscInt i,PetscReal u,PetscInt p,PetscInt d,const PetscReal U[],PetscReal L[]);
extern void IGA_Basis_Spectral(PetscInt i,PetscReal u,PetscInt p,PetscInt d,const PetscReal U[],PetscReal L[]);
EXTERN_C_END

PetscErrorCode IGABasisInitQuadrature(IGABasis basis,IGAAxis axis,IGARule rule)
{
  PetscInt       p,m,n;
  const PetscReal*U;
  PetscInt       iel,nel;
  PetscInt       iqp,nqp;
  PetscInt       nen,ndr,d;
  PetscInt       *offset;
  PetscReal      *detJac;
  PetscReal      *weight;
  PetscReal      *point;
  PetscReal      *value;
  void          (*ComputeBasis)(PetscInt,PetscReal,PetscInt,PetscInt,const PetscReal[],PetscReal[]) = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(basis,1);
  PetscValidPointer(axis,2);
  PetscValidPointer(rule,3);

  if (rule->nqp < 1) {ierr = IGARuleSetSize(rule,axis->p+1);CHKERRQ(ierr);}

  p = axis->p;
  m = axis->m;
  n = m - p - 1;
  U = axis->U;

  if (basis->type != IGA_BASIS_BSPLINE) {
    PetscInt k,j,s;
    for (k=1,j=m; k<m; k=j) { /* check multiplicity */
      j = IGA_NextKnot(m,U,k,1);
      if ((s = j-k) < p)
        SETERRQ6(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
                 "Basis type %s requires C^0 continuity, "
                 "Knot U[%D:%D]=%g has multiplicity %D "
                 "less than polynomial degree %D",
                 IGABasisTypes[basis->type],k,j-1,(double)U[k],s,p);
    }
  }

  /* Compute quadrature points and weights */
  nel = axis->nel;
  nqp = rule->nqp;
  ierr = PetscMalloc1((size_t)nel,&detJac);CHKERRQ(ierr);
  ierr = PetscMalloc1((size_t)(nel*nqp),&weight);CHKERRQ(ierr);
  ierr = PetscMalloc1((size_t)(nel*nqp),&point);CHKERRQ(ierr);
  if (rule->type != IGA_RULE_REDUCED) {
    const PetscReal *X = rule->point;
    const PetscReal *W = rule->weight;
    for (iel=0; iel<nel; iel++) {
      PetscInt  k  = axis->span[iel];
      PetscReal u0 = U[k], u1 = U[k+1];
      PetscReal J  = (u1-u0)/2;
      PetscReal *w = weight + iel*nqp;
      PetscReal *u = point  + iel*nqp;
      detJac[iel] = J;
      for (iqp=0; iqp<nqp; iqp++) {
        w[iqp] = W[iqp];
        u[iqp] = (X[iqp] + 1) * J + u0;
      }
    }
  } else {
#define SetRule(e)                       \
    do {                                 \
      const PetscInt  nr = rule->nqp;    \
      const PetscReal *X = rule->point;  \
      const PetscReal *W = rule->weight; \
      PetscInt  k  = axis->span[(e)];    \
      PetscReal u0 = U[k], u1 = U[k+1];  \
      PetscReal J  = (u1-u0)/2;          \
      PetscReal *w = weight + (e)*nqp;   \
      PetscReal *u = point  + (e)*nqp;   \
      detJac[(e)] = J;                   \
      for (iqp=0; iqp<nr; iqp++) {       \
        w[iqp] = W[iqp];                 \
        u[iqp] = (X[iqp] + 1) * J + u0;  \
      }                                  \
      for (iqp=nr; iqp<nqp; iqp++) {     \
        w[iqp] = 0;                      \
        u[iqp] = PETSC_MAX_REAL;         \
      }                                  \
    } while (0)
    /* */
    if (nel > 0) SetRule(0);     /* first  */
    if (nel > 1) SetRule(nel-1); /* last   */
    if (nel > 2) {               /* others */
      if (nqp > 1) {ierr = IGARuleSetSize(rule,nqp-1);CHKERRQ(ierr);}
      for (iel=1; iel<nel-1; iel++) SetRule(iel);
    }
  }

  /* Compute basis functions and derivatives */
  nen = p+1;
  ndr = 5;
  d = PetscMin(p,4);
  switch (basis->type) {
  case IGA_BASIS_BSPLINE:
  case IGA_BASIS_BERNSTEIN:
    ComputeBasis = IGA_Basis_BSpline; break;
  case IGA_BASIS_LAGRANGE:
    ComputeBasis = IGA_Basis_Lagrange; break;
  case IGA_BASIS_SPECTRAL:
    ComputeBasis = IGA_Basis_Spectral; break;
  }
  ierr = PetscMalloc1((size_t)nel,&offset);CHKERRQ(ierr);
  ierr = PetscMalloc1((size_t)(nel*nqp*nen*ndr),&value);CHKERRQ(ierr);
  for (iel=0; iel<nel; iel++) {
    PetscInt  k  = axis->span[iel];
    PetscReal *w = weight + iel*nqp;
    PetscReal *u = point  + iel*nqp;
    PetscReal *N = value  + iel*nqp*nen*ndr;
    offset[iel] = k - p;
    for (iqp=0; iqp<nqp && w[iqp]>0; iqp++)
      ComputeBasis(k,u[iqp],p,d,U,&N[iqp*nen*ndr]);
  }

  ierr = IGABasisReset(basis);CHKERRQ(ierr);
  basis->nel    = nel;
  basis->nqp    = nqp;
  basis->nen    = nen;
  basis->offset = offset;
  basis->detJac = detJac;
  basis->weight = weight;
  basis->point  = point;
  basis->value  = value;
  {
    PetscInt  k0 = p,     k1 = n;
    PetscReal u0 = U[k0], u1 = U[k1+1];
    ierr = PetscMalloc1((size_t)(nen*ndr),&basis->bnd_value[0]);CHKERRQ(ierr);
    ierr = PetscMalloc1((size_t)(nen*ndr),&basis->bnd_value[1]);CHKERRQ(ierr);
    basis->bnd_point[0] = u0; basis->bnd_detJac = 1.0;
    basis->bnd_point[1] = u1; basis->bnd_weight = 1.0;
    ComputeBasis(k0,u0,p,d,U,basis->bnd_value[0]);
    ComputeBasis(k1,u1,p,d,U,basis->bnd_value[1]);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
extern PetscInt  IGA_FindSpan(PetscInt n,PetscInt p,PetscReal u, const PetscReal *U);
extern PetscReal IGA_Greville(PetscInt i,PetscInt p,const PetscReal U[]);
EXTERN_C_END

PetscErrorCode IGABasisInitCollocation(IGABasis basis,IGAAxis axis)
{
  PetscInt       p,m,n;
  const PetscReal*U;
  PetscInt       inp,nnp,shift;
  PetscInt       nen,ndr,d;
  PetscInt       *offset;
  PetscReal      *detJac;
  PetscReal      *weight;
  PetscReal      *point;
  PetscReal      *value;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(basis,1);
  PetscValidPointer(axis,2);

  p = axis->p;
  m = axis->m;
  n = m - p - 1;
  U = axis->U;

  if (basis->type != IGA_BASIS_BSPLINE)
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
             "Basis type is %s, collocation method requires %s",
             IGABasisTypes[basis->type],IGABasisTypes[IGA_BASIS_BSPLINE]);

  nnp = axis->nnp;
  nen = p+1;
  ndr = 5;
  d   = PetscMin(p,4);

  shift = (n + 1 - nnp)/2;

  ierr = PetscMalloc1((size_t)nnp,&offset);CHKERRQ(ierr);
  ierr = PetscMalloc1((size_t)nnp,&detJac);CHKERRQ(ierr);
  ierr = PetscMalloc1((size_t)nnp,&weight);CHKERRQ(ierr);
  ierr = PetscMalloc1((size_t)nnp,&point);CHKERRQ(ierr);
  ierr = PetscMalloc1((size_t)(nnp*nen*ndr),&value);CHKERRQ(ierr);

  for (inp=0; inp<nnp; inp++) {
    PetscReal u = IGA_Greville(inp+shift,p,U);
    PetscInt  k = IGA_FindSpan(n,p,u,U);
    PetscReal *N = &value[inp*nen*ndr];
    offset[inp] = k-p-shift;
    detJac[inp] = 1.0;
    weight[inp] = 1.0;
    point[inp]  = u;
    IGA_Basis_BSpline(k,u,p,d,U,N);
  }

  ierr = IGABasisReset(basis);CHKERRQ(ierr);

  basis->nel    = nnp;
  basis->nqp    = 1;
  basis->nen    = nen;
  basis->offset = offset;
  basis->detJac = detJac;
  basis->weight = weight;
  basis->point  = point;
  basis->value  = value;

  {
    PetscInt  k0 = p,    k1 = n;
    PetscReal u0 = U[p], u1 = U[n+1];
    ierr = PetscMalloc1((size_t)(nen*ndr),&basis->bnd_value[0]);CHKERRQ(ierr);
    ierr = PetscMalloc1((size_t)(nen*ndr),&basis->bnd_value[1]);CHKERRQ(ierr);
    basis->bnd_point[0] = u0; basis->bnd_detJac = 1.0;
    basis->bnd_point[1] = u1; basis->bnd_weight = 1.0;
    IGA_Basis_BSpline(k0,u0,p,d,U,basis->bnd_value[0]);
    IGA_Basis_BSpline(k1,u1,p,d,U,basis->bnd_value[1]);
  }
  PetscFunctionReturn(0);
}

PetscInt IGA_FindSpan(PetscInt n,PetscInt p,PetscReal u, const PetscReal U[])
{
  PetscInt low,high,span;
  if (PetscUnlikely(u <= U[p]))   return p;
  if (PetscUnlikely(u >= U[n+1])) return n;
  low  = p;
  high = n+1;
  span = (high+low)/2;
  while (u < U[span] || u >= U[span+1]) {
    if (u < U[span]) {
      high = span;
    } else {
      low = span;
    }
    span = (high+low)/2;
  }
  return span;
}

PetscReal IGA_Greville(PetscInt i,PetscInt p,const PetscReal U[])
{
  PetscInt j;
  PetscReal u = 0.0;
  for (j=0; j<p; j++) u += U[i+j+1];
  return PetscLikely(p>0) ? u/(PetscReal)p : (U[0]+U[1])/2;
}
