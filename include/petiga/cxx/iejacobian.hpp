#ifndef PETIGA_CXX_IEJACOBIAN_HPP
#define PETIGA_CXX_IEJACOBIAN_HPP

namespace {
template <int dim, int nen, int dof>
PetscErrorCode IEJacobian(IGAPoint q,PetscReal s,const PetscScalar V[],PetscReal t,const PetscScalar U[],PetscReal t0,const PetscScalar U0[],PetscScalar J[],void *ctx)
{
  typedef const PetscScalar (&ArrayI)[nen][dof];
  typedef /* */ PetscScalar (&ArrayO)[nen][dof][nen][dof];
  ArrayI arrayV  = reinterpret_cast<ArrayI>(*V);
  ArrayI arrayU  = reinterpret_cast<ArrayI>(*U);
  ArrayI arrayU0 = reinterpret_cast<ArrayI>(*U0);
  ArrayO arrayJ  = reinterpret_cast<ArrayO>(*J);
  return IJacobian<dim>(q,s,arrayV,t,arrayU,t0,arrayU0,arrayJ,ctx);
}
}

#include "lookup.hpp"

extern "C"
PetscErrorCode IEJacobianCXX(IGAPoint q,PetscReal s,const PetscScalar V[],PetscReal t,const PetscScalar U[],PetscReal t0,const PetscScalar U0[],PetscScalar F[],void *ctx)
{
  IGAFormIEJacobian IEJacobianP = NULL;
  LookupTemplateSet(IEJacobianP,q,IEJacobian);
  LookupTemplateChk(IEJacobianP,q,IEJacobian);
  return IEJacobianP(q,s,V,t,U,J,ctx);
}

#endif
