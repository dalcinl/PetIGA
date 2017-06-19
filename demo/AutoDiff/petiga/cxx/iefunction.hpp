#ifndef PETIGA_CXX_IEFUNCTION_HPP
#define PETIGA_CXX_IEFUNCTION_HPP

namespace {
template <int dim, int nen, int dof>
PetscErrorCode IEFunction(IGAPoint q,PetscReal s,const PetscScalar V[],PetscReal t,const PetscScalar U[],PetscReal t0,const PetscScalar U0[],PetscScalar F[],void *ctx)
{
  typedef const PetscScalar (&ArrayI)[nen][dof];
  typedef /* */ PetscScalar (&ArrayO)[nen][dof];
  ArrayI arrayV  = reinterpret_cast<ArrayI>(*V);
  ArrayI arrayU  = reinterpret_cast<ArrayI>(*U);
  ArrayI arrayU0 = reinterpret_cast<ArrayI>(*U0);
  ArrayO arrayF  = reinterpret_cast<ArrayO>(*F);
  return IFunction<dim>(q,s,arrayV,t,arrayU,t0,arrayU0,arrayF,ctx);
}
}

#include "lookup.hpp"

extern "C"
PetscErrorCode IEFunctionCXX(IGAPoint q,PetscReal s,const PetscScalar V[],PetscReal t,const PetscScalar U[],PetscReal t0,const PetscScalar U0[],PetscScalar F[],void *ctx)
{
  IGAFormIEFunction IEFunctionP = NULL;
  LookupTemplateSet(IEFunctionP,q,IEFunction);
  LookupTemplateChk(IEFunctionP,q,IEFunction);
  return IEFunctionP(q,s,V,t,U,F,ctx);
}

#endif
