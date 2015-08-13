#ifndef PETIGA_FAD_JACOBIAN_HPP
#define PETIGA_FAD_JACOBIAN_HPP

#include "adtype.hpp"

namespace {
template <int dim, int nen, int dof, typename Scalar>
PetscErrorCode Jacobian(const IGAPoint q,
                        const Scalar (&U)[nen][dof],
                        /* */ Scalar (&J)[nen][dof][nen][dof],
                        void *ctx)
{
  typedef PetIGA::AD<Scalar,nen,dof> AD;
  typedef typename AD::Type ADType;
  ADType X[nen][dof]; AD::Input(X, U);
  ADType R[nen][dof];//AD::Clear(R);
  Function<dim,nen,dof,ADType>(q, X, R, ctx);
  AD::Output(R, J);
  return 0;
}
}

#endif
