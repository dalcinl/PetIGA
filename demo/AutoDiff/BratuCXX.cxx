#include "petiga/petiga.hpp"

typedef struct {
  PetscReal lambda;
} Params;

template <int dim, int nen, int dof, typename Scalar>
static inline
PetscErrorCode Function(IGAPoint q,
                        const Scalar (&U)[nen][dof],
                        /* */ Scalar (&F)[nen][dof],
                        void *ctx)
{
  using namespace PetIGA;
  typedef  Space<dim,nen> Space;
  typename Space::Shape0 N0 = Space::GetShape0(q);
  typename Space::Shape1 N1 = Space::GetShape1(q);
  PetscReal lambda = ((Params*)ctx)->lambda;

  Scalar u,grad_u[dim];
  Point::FormValue(q, U, u);
  Point::FormGrad (q, U, grad_u);

  Scalar lambda_exp_u = lambda*exp(u);

  for (int a=0; a<nen; a++)
    F[a][0] = dot(N1[a],grad_u) - lambda_exp_u * N0[a];

  return 0;
}

template <int dim, int nen, int dof, typename Scalar>
static inline
PetscErrorCode Jacobian(IGAPoint q,
                        const Scalar (&U)[nen][dof],
                        /* */ Scalar (&J)[nen][dof][nen][dof],
                        void *ctx)

{
  using namespace PetIGA;
  typedef  Space<dim,nen> Space;
  typename Space::Shape0 N0 = Space::GetShape0(q);
  typename Space::Shape1 N1 = Space::GetShape1(q);
  PetscReal lambda = ((Params*)ctx)->lambda;

  Scalar u;
  Point::FormValue(q, U, u);

  Scalar lambda_exp_u = lambda*exp(u);

  for (int a=0; a<nen; a++)
    for (int b=0; b<nen; b++)
      J[a][0][b][0] = dot(N1[a],N1[b]) - lambda_exp_u * N0[a]*N0[b];

  return 0;
}

#define DOF 1
#include "petiga/cxx/function.hpp"
#include "petiga/cxx/jacobian.hpp"
