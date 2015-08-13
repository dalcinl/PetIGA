#include "petiga/petiga.hpp"
#include "CahnHilliard.hpp"

template<int dim, int nen, int dof, typename Scalar>
static inline
PetscErrorCode IFunction(IGAPoint  q,
                         PetscReal _,const Scalar (&V)[nen][dof],
                         PetscReal t,const Scalar (&U)[nen][dof],
                         Scalar (&F)[nen][dof],
                         void *ctx)
{
  const Params *params = (Params *)ctx;

  using namespace PetIGA;
  typedef  Space<dim,nen> Space;
  typename Space::Shape0 N0 = Space::GetShape0(q);
  typename Space::Shape1 N1 = Space::GetShape1(q);

  Scalar c_t;
  Point::FormValue<0>(q, V, c_t);
  Scalar u[2],grad_u[2][dim];
  Point::FormValue(q, U, u);
  Point::FormGrad (q, U, grad_u);
  Scalar& c = u[0];
  Scalar& s = u[1];
  Scalar (&grad_c)[dim] = grad_u[0];
  Scalar (&grad_s)[dim] = grad_u[1];

  Scalar M;
  Mobility(params,c,M);
  Scalar mu;
  ChemicalPotential(params,c,mu);

  for (int a=0; a<nen; a++) {
    F[a][0] = N0[a] *  c_t   + dot(N1[a],grad_s) * M;
    F[a][1] = N0[a] * (s-mu) - dot(N1[a],grad_c);
  }

  return 0;
}

template<int dim, int nen, int dof, typename Scalar>
static inline
PetscErrorCode IJacobian(IGAPoint  q,
                         PetscReal shift, const Scalar (&V)[nen][dof],
                         PetscReal,       const Scalar (&U)[nen][dof],
                         Scalar (&J)[nen][dof][nen][dof],
                         void *ctx)
{
  const Params *params = (Params *)ctx;

  using namespace PetIGA;
  typedef  Space<dim,nen> Space;
  typename Space::Shape0 N0 = Space::GetShape0(q);
  typename Space::Shape1 N1 = Space::GetShape1(q);

  Scalar c, grad_s[dim];
  Point::FormValue<0>(q,U,c);
  Point::FormGrad <1>(q,U,grad_s);

  Scalar M,dM;
  Mobility(params,c,M,dM);
  Scalar mu,dmu;
  ChemicalPotential(params,c,mu,dmu);

  for (int a=0; a<nen; a++) {
    Scalar dNa_dot_gs_dM = dot(N1[a],grad_s) * dM;
    for (int b=0; b<nen; b++) {
      Scalar gNa_dot_gNb = dot(N1[a],N1[b]);
      J[a][0][b][0] += shift * N0[a] * N0[b] + dNa_dot_gs_dM * N0[b];
      J[a][0][b][1] += M * gNa_dot_gNb;
      J[a][1][b][0] += - dmu * N0[a] * N0[b] - gNa_dot_gNb;
      J[a][1][b][1] += N0[a] * N0[b];
    }
  }

  return 0;
}

#define DOF 2
#include "petiga/cxx/ifunction.hpp"
#include "petiga/cxx/ijacobian.hpp"
