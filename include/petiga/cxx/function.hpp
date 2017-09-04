#ifndef PETIGA_CXX_FUNCTION_HPP
#define PETIGA_CXX_FUNCTION_HPP

namespace {
template <int dim, int nen, int dof>
PetscErrorCode Function(const IGAPoint q,const PetscScalar U[],PetscScalar F[],void *ctx)
{
  typedef const PetscScalar (&ArrayI)[nen][dof];
  typedef       PetscScalar (&ArrayO)[nen][dof];
  ArrayI arrayU = reinterpret_cast<ArrayI>(*U);
  ArrayO arrayF = reinterpret_cast<ArrayO>(*F);
  return Function<dim>(q,arrayU,arrayF,ctx);
}
}

#include "lookup.hpp"

extern "C"
PetscErrorCode FunctionCXX(IGAPoint q,const PetscScalar U[],PetscScalar F[],void *ctx)
{
  IGAFormFunction FunctionP = NULL;
  LookupTemplateSet(FunctionP,q,Function);
  LookupTemplateChk(FunctionP,q,Function);
  return FunctionP(q,U,F,ctx);
}

#endif
