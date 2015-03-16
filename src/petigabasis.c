#include "petiga.h"

const char *const IGABasisTypes[] = {
  "BSPLINE",
  "BERNSTEIN",
  "LAGRANGE",
  "SPECTRAL",
  /* */
  "IGABasisType","IGA_BASIS_",NULL};

#undef  __FUNCT__
#define __FUNCT__ "IGABasisCreate"
PetscErrorCode IGABasisCreate(IGABasis *basis)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(basis,3);
  ierr = PetscCalloc1(1,basis);CHKERRQ(ierr);
  (*basis)->refct = 1;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGABasisDestroy"
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

#undef  __FUNCT__
#define __FUNCT__ "IGABasisReset"
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

#undef  __FUNCT__
#define __FUNCT__ "IGABasisReference"
PetscErrorCode IGABasisReference(IGABasis basis)
{
  PetscFunctionBegin;
  PetscValidPointer(basis,1);
  basis->refct++;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGABasisSetType"
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
extern void IGA_Basis_BSpline (PetscInt i,PetscReal u,PetscInt p,PetscInt d,const PetscReal U[],PetscReal B[]);
extern void IGA_Basis_Lagrange(PetscInt i,PetscReal u,PetscInt p,PetscInt d,const PetscReal U[],PetscReal L[]);
extern void IGA_Basis_Spectral(PetscInt i,PetscReal u,PetscInt p,PetscInt d,const PetscReal U[],PetscReal L[]);
EXTERN_C_END

#undef  __FUNCT__
#define __FUNCT__ "IGABasisInitQuadrature"
PetscErrorCode IGABasisInitQuadrature(IGABasis basis,IGAAxis axis,IGARule rule)
{
  PetscInt       p,m,n;
  const PetscInt *span;
  const PetscReal*U,*X,*W;
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

  p = axis->p;
  m = axis->m;
  n = m - p - 1;
  U = axis->U;

  if (basis->type != IGA_BASIS_BSPLINE) {
    PetscInt s,j,k=1;
    while (k < m) {
      j = k; s = 1; while (++k < m && U[j] == U[k]) s++;
      if (s < p) SETERRQ5(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
                          "Basis type %s requires C^0 continuity, "
                          "Knot U[%D]=%g has multiplicity %D "
                          "less than polynomial degree %D",
                          IGABasisTypes[basis->type],j,(double)U[j],s,p);
    }
  }

  nel  = axis->nel;
  span = axis->span;

  if (rule->nqp < 1) {ierr = IGARuleInit(rule,p+1);CHKERRQ(ierr);}
  nqp = rule->nqp;
  X = rule->point;
  W = rule->weight;
  ierr = PetscMalloc1((size_t)nel,&detJac);CHKERRQ(ierr);
  ierr = PetscMalloc1((size_t)(nel*nqp),&weight);CHKERRQ(ierr);
  ierr = PetscMalloc1((size_t)(nel*nqp),&point);CHKERRQ(ierr);
  for (iel=0; iel<nel; iel++) {
    PetscInt  k  = span[iel];
    PetscReal u0 = U[k], u1 = U[k+1];
    PetscReal J  = (u1-u0)/2;
    PetscReal *w = &weight[iel*nqp];
    PetscReal *u = &point[iel*nqp];
    detJac[iel] = J;
    for (iqp=0; iqp<nqp; iqp++) {
      w[iqp] = W[iqp];
      u[iqp] = (X[iqp] + 1) * J + u0;
    }
  }

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
    PetscInt  k  = span[iel];
    PetscReal *w = &weight[iel*nqp];
    PetscReal *u = &point[iel*nqp];
    PetscReal *N = &value[iel*nqp*nen*ndr];
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

#undef  __FUNCT__
#define __FUNCT__ "IGABasisInitCollocation"
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
