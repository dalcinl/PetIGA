#include "petiga.h"

#undef  __FUNCT__
#define __FUNCT__ "IGABasisCreate"
PetscErrorCode IGABasisCreate(IGABasis *basis)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(basis,3);
  ierr = PetscNew(struct _n_IGABasis,basis);CHKERRQ(ierr);
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
  basis = *_basis; *_basis = 0;
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
  basis->p   = 0;
  basis->d   = 0;
  ierr = PetscFree(basis->offset);CHKERRQ(ierr);
  ierr = PetscFree(basis->detJ);CHKERRQ(ierr);
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

EXTERN_C_BEGIN
extern void IGA_Basis_BSpline(PetscInt i,PetscReal u,PetscInt p,PetscInt d,const PetscReal U[],PetscReal N[]);
EXTERN_C_END

#undef  __FUNCT__
#define __FUNCT__ "IGABasisInitQuadrature"
PetscErrorCode IGABasisInitQuadrature(IGABasis basis,IGAAxis axis,IGARule rule,PetscInt d)
{
  PetscInt       p;
  const PetscInt *span;
  const PetscReal*U,*X,*W;
  PetscInt       iel,nel;
  PetscInt       iqp,nqp;
  PetscInt       nen,ndr;
  PetscInt       *offset;
  PetscReal      *detJ;
  PetscReal      *weight;
  PetscReal      *point;
  PetscReal      *value;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(basis,1);
  PetscValidPointer(axis,2);
  PetscValidPointer(rule,3);
  if (d < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
                      "Derivative order must be grather than zero, got %D",d);

  p = axis->p;
  U = axis->U;

  nqp = rule->nqp;
  X   = rule->point;
  W   = rule->weight;

  nel  = axis->nel;
  span = axis->span;
  nen  = p+1;
  ndr  = d+1;

  ierr = PetscMalloc1(nel,PetscInt,&offset);CHKERRQ(ierr);
  ierr = PetscMalloc1(nel,PetscReal,&detJ);CHKERRQ(ierr);
  ierr = PetscMalloc1(nqp,PetscReal,&weight);CHKERRQ(ierr);
  ierr = PetscMalloc1(nel*nqp,PetscReal,&point);CHKERRQ(ierr);
  ierr = PetscMalloc1(nel*nqp*nen*ndr,PetscReal,&value);CHKERRQ(ierr);

  for (iqp=0; iqp<nqp; iqp++) {
    weight[iqp] = W[iqp];
  }
  for (iel=0; iel<nel; iel++) {
    PetscInt  k = span[iel];
    PetscReal u0 = U[k], u1 = U[k+1];
    PetscReal J = (u1-u0)/2;
    PetscReal *u = &point[iel*nqp];
    PetscReal *N = &value[iel*nqp*nen*ndr];
    detJ[iel] = J;
    for (iqp=0; iqp<nqp; iqp++) {
      u[iqp] = (X[iqp] + 1) * J + u0;
      IGA_Basis_BSpline(k,u[iqp],p,d,U,&N[iqp*nen*ndr]);
    }
    offset[iel] = k-p;
  }

  ierr = IGABasisReset(basis);CHKERRQ(ierr);

  basis->nel    = nel;
  basis->nqp    = nqp;
  basis->nen    = nen;
  basis->p      = p;
  basis->d      = d;
  basis->offset = offset;

  basis->detJ   = detJ;
  basis->weight = weight;
  basis->point  = point;
  basis->value  = value;

  {
    PetscInt  o0 = offset[0], o1 = offset[nel-1];
    PetscInt  k0 = span[0],   k1 = span[nel-1];
    PetscReal u0 = U[k0],     u1 = U[k1+1];
    ierr = PetscMalloc1(nen*ndr,PetscReal,&basis->bnd_value[0]);CHKERRQ(ierr);
    ierr = PetscMalloc1(nen*ndr,PetscReal,&basis->bnd_value[1]);CHKERRQ(ierr);
    basis->bnd_offset[0] =  o0; basis->bnd_offset[1] =  o1;
    basis->bnd_detJ  [0] = 1.0; basis->bnd_detJ  [1] = 1.0;
    basis->bnd_weight[0] = 1.0; basis->bnd_weight[1] = 1.0;
    basis->bnd_point [0] =  u0; basis->bnd_point [1] =  u1;
    IGA_Basis_BSpline(k0,u0,p,d,U,basis->bnd_value[0]);
    IGA_Basis_BSpline(k1,u1,p,d,U,basis->bnd_value[1]);
  }

  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
extern PetscInt  IGA_FindSpan(PetscInt n,PetscInt p,PetscReal u, const PetscReal *U);
extern PetscReal IGA_Greville(PetscInt i,PetscInt p,const PetscReal U[]);
EXTERN_C_END

#undef  __FUNCT__
#define __FUNCT__ "IGABasisInitCollocation"
PetscErrorCode IGABasisInitCollocation(IGABasis basis,IGAAxis axis,PetscInt d)
{
  PetscInt       p,n;
  const PetscReal*U;
  PetscInt       iel,nel;
  PetscInt       iqp,nqp;
  PetscInt       nen,ndr;
  PetscInt       *offset;
  PetscReal      *detJ;
  PetscReal      *weight;
  PetscReal      *point;
  PetscReal      *value;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(basis,1);
  PetscValidPointer(axis,2);
  if (d < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
                      "Derivative order must be grather than zero, got %D",d);

  p = axis->p;
  n = axis->m - p -1;
  U = axis->U;

  nel  = axis->nnp;
  nqp  = 1;
  nen  = p+1;
  ndr  = d+1;

  ierr = PetscMalloc1(nel,PetscInt,&offset);CHKERRQ(ierr);
  ierr = PetscMalloc1(nel,PetscReal,&detJ);CHKERRQ(ierr);
  ierr = PetscMalloc1(nqp,PetscReal,&weight);CHKERRQ(ierr);
  ierr = PetscMalloc1(nel*nqp,PetscReal,&point);CHKERRQ(ierr);
  ierr = PetscMalloc1(nel*nqp*nen*ndr,PetscReal,&value);CHKERRQ(ierr);

  for (iqp=0; iqp<nqp; iqp++) {
    weight[iqp] = 1.0;
  }
  for (iel=0; iel<nel; iel++) {
    PetscReal u = IGA_Greville(iel,p,U);
    PetscInt  k = IGA_FindSpan(n,p,u,U);
    PetscReal *N = &value[iel*nen*ndr];
    offset[iel] = k-p;
    point[iel]  = u;
    detJ[iel]   = U[k+1]-U[k];
    IGA_Basis_BSpline(k,u,p,d,U,N);
  }

  ierr = IGABasisReset(basis);CHKERRQ(ierr);

  basis->nel    = nel;
  basis->nqp    = nqp;
  basis->nen    = nen;
  basis->p      = p;
  basis->d      = d;
  basis->offset = offset;

  basis->detJ   = detJ;
  basis->weight = weight;
  basis->point  = point;
  basis->value  = value;

  PetscFunctionReturn(0);
}

PetscInt IGA_FindSpan(PetscInt n,PetscInt p,PetscReal u, const PetscReal U[])
{
  PetscInt low,high,span;
  if(u >= U[n+1]) return n;
  if(u <= U[p])   return p;
  low  = p;
  high = n+1;
  span = (high+low)/2;
  while(u < U[span] || u >= U[span+1]){
    if(u < U[span])
      high = span;
    else
      low = span;
    span = (high+low)/2;
  }
  return span;
}

PetscReal IGA_Greville(PetscInt i,PetscInt p,const PetscReal U[])
{
  PetscInt j;
  PetscReal u = 0.0;
  for(j=0;j<p;j++) u += U[i+j+1];
  u /= p;
  return u;
}
