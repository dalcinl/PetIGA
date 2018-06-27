#ifndef PETIGA_CXX_SYSTEM_HPP
#define PETIGA_CXX_SYSTEM_HPP

namespace {
template <int dim, int nen, int dof>
PetscErrorCode System(const IGAPoint q,PetscScalar K[],PetscScalar F[],void *ctx)
{
  typedef PetscScalar (&ArrayK)[nen][dof][nen][dof];
  typedef PetscScalar (&ArrayF)[nen][dof];
  ArrayK arrayK = reinterpret_cast<ArrayK>(*K);
  ArrayF arrayF = reinterpret_cast<ArrayF>(*F);
  return System<dim>(q,arrayK,arrayF,ctx);
}
}

#include "lookup.hpp"

extern "C"
PetscErrorCode SystemCXX(IGAPoint q,PetscScalar K[],PetscScalar F[],void *ctx)
{
  IGAFormSystem SystemP = NULL;
  LookupTemplateSet(SystemP,q,System);
  LookupTemplateChk(SystemP,q,System);
  return SystemP(q,K,F,ctx);
}

#endif
