#ifndef PETIGA_CXX_JACOBIAN_HPP
#define PETIGA_CXX_JACOBIAN_HPP

namespace {
template <int dim, int nen, int dof>
PetscErrorCode Jacobian(const IGAPoint q,const PetscScalar U[],PetscScalar J[],void *ctx)
{
  typedef const PetscScalar (&ArrayI)[nen][dof];
  typedef       PetscScalar (&ArrayO)[nen][dof][nen][dof];
  ArrayI arrayU = reinterpret_cast<ArrayI>(*U);
  ArrayO arrayJ = reinterpret_cast<ArrayO>(*J);
  return Jacobian<dim>(q,arrayU,arrayJ,ctx);
}
}

#include "lookup.hpp"

extern "C"
PetscErrorCode JacobianCXX(IGAPoint q,const PetscScalar U[],PetscScalar J[],void *ctx)
{
  IGAFormJacobian JacobianP = NULL;
  LookupTemplateSet(JacobianP,q,Jacobian);
  LookupTemplateChk(JacobianP,q,Jacobian);
  return JacobianP(q,U,J,ctx);
}

#endif
