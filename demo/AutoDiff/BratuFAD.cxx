#include "petiga/petiga.hpp"

typedef struct {
  PetscReal lambda;
} Params;

template<int dim, int nen, int dof, typename Scalar>
static inline
PetscErrorCode Function(const IGAPoint q,
                        const Scalar (&U)[nen][dof],
                        /* */ Scalar (&F)[nen][dof],
                        void *ctx)
{
  using namespace PetIGA;
  typedef  Space<dim,nen> Space;
  typename Space::Shape0 N0 = Space::GetShape0(q);
  typename Space::Shape1 N1 = Space::GetShape1(q);
  const PetscReal lambda    = ((Params*)ctx)->lambda;

  Scalar u, grad_u[dim];
  Point::FormValue(q, U, u);
  Point::FormGrad (q, U, grad_u);

  Scalar lambda_exp_u = lambda*exp(u);

  for (int a=0; a<nen; a++)
    F[a][0] = dot(N1[a],grad_u) - N0[a]*lambda_exp_u;

  return 0;
}


#include "petiga/fad/jacobian.hpp"
#define DOF 1
#define FunctionCXX FunctionFAD
#define JacobianCXX JacobianFAD
#include "petiga/cxx/function.hpp"
#include "petiga/cxx/jacobian.hpp"
