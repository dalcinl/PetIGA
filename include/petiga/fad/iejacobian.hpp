#ifndef PETIGA_FAD_IEJACOBIAN_HPP
#define PETIGA_FAD_IEJACOBIAN_HPP

#include "adtype.hpp"

namespace {
template <int dim, int nen, int dof, typename Scalar>
PetscErrorCode IEJacobian(IGAPoint  q,
                          PetscReal s, const Scalar (&V )[nen][dof],
                          PetscReal t, const Scalar (&U )[nen][dof],
                          PetscReal t0,const Scalar (&U0)[nen][dof],
                          Scalar (&J)[nen][dof][nen][dof],
                          void *ctx)
{
  typedef PetIGA::AD<Scalar,nen,dof> AD;
  typedef typename AD::Type ADType;
  ADType Y[nen][dof]; AD::Input(Y, V, s);
  ADType X[nen][dof]; AD::Input(X, U);
  ADType R[nen][dof];//AD::Clear(R);
  IEFunction<dim,nen,dof>(q, s, Y, t, X, t0, U0, R, ctx);
  AD::Output(R, J);
  return 0;
}
}

#endif
