#ifndef PETIGA_FAD_IJACOBIAN_HPP
#define PETIGA_FAD_IJACOBIAN_HPP

#include "adtype.hpp"

namespace {
template <int dim, int nen, int dof, typename Scalar>
PetscErrorCode IJacobian(IGAPoint  q,
                         PetscReal s, const Scalar (&V)[nen][dof],
                         PetscReal t, const Scalar (&U)[nen][dof],
                         Scalar (&J)[nen][dof][nen][dof],
                         void *ctx)
{
  typedef PetIGA::AD<Scalar,nen,dof> AD;
  typedef typename AD::Type ADType;
  ADType Y[nen][dof]; AD::Input(Y, V, s);
  ADType X[nen][dof]; AD::Input(X, U);
  ADType R[nen][dof];//AD::Clear(R);
  IFunction<dim,nen,dof,ADType>(q, s, Y, t, X, R, ctx);
  AD::Output(R, J);
  return 0;
}
}

#endif
