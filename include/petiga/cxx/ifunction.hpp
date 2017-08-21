#ifndef PETIGA_CXX_IFUNCTION_HPP
#define PETIGA_CXX_IFUNCTION_HPP

namespace {
template <int dim, int nen, int dof>
PetscErrorCode IFunction(IGAPoint q,PetscReal s,const PetscScalar V[],PetscReal t,const PetscScalar U[],PetscScalar F[],void *ctx)
{
  typedef const PetscScalar (&ArrayI)[nen][dof];
  typedef /* */ PetscScalar (&ArrayO)[nen][dof];
  ArrayI arrayV = reinterpret_cast<ArrayI>(*V);
  ArrayI arrayU = reinterpret_cast<ArrayI>(*U);
  ArrayO arrayF = reinterpret_cast<ArrayO>(*F);
  return IFunction<dim>(q,s,arrayV,t,arrayU,arrayF,ctx);
}
}

#include "lookup.hpp"

extern "C"
PetscErrorCode IFunctionCXX(IGAPoint q,PetscReal s,const PetscScalar V[],PetscReal t,const PetscScalar U[],PetscScalar F[],void *ctx)
{
  IGAFormIFunction IFunctionP = NULL;
  LookupTemplateSet(IFunctionP,q,IFunction);
  LookupTemplateChk(IFunctionP,q,IFunction);
  return IFunctionP(q,s,V,t,U,F,ctx);
}

#endif
