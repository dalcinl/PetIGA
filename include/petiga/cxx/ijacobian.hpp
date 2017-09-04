#ifndef PETIGA_CXX_IJACOBIAN_HPP
#define PETIGA_CXX_IJACOBIAN_HPP

namespace {
template <int dim, int nen, int dof>
PetscErrorCode IJacobian(IGAPoint q,PetscReal s,const PetscScalar V[],PetscReal t,const PetscScalar U[],PetscScalar J[],void *ctx)
{
  typedef const PetscScalar (&ArrayI)[nen][dof];
  typedef /* */ PetscScalar (&ArrayO)[nen][dof][nen][dof];
  ArrayI arrayV = reinterpret_cast<ArrayI>(*V);
  ArrayI arrayU = reinterpret_cast<ArrayI>(*U);
  ArrayO arrayJ = reinterpret_cast<ArrayO>(*J);
  return IJacobian<dim>(q,s,arrayV,t,arrayU,arrayJ,ctx);
}
}

#include "lookup.hpp"

extern "C"
PetscErrorCode IJacobianCXX(IGAPoint q,PetscReal s,const PetscScalar V[],PetscReal t,const PetscScalar U[],PetscScalar J[],void *ctx)
{
  IGAFormIJacobian IJacobianP = NULL;
  LookupTemplateSet(IJacobianP,q,IJacobian);
  LookupTemplateChk(IJacobianP,q,IJacobian);
  return IJacobianP(q,s,V,t,U,J,ctx);
}

#endif
