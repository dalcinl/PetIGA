#include "petiga/petiga.hpp"
#include "CahnHilliard.hpp"

template <int dim, int nen, int dof, typename Scalar>
static inline
PetscErrorCode IFunction(IGAPoint q,
                         PetscReal ,const Scalar (&V)[nen][dof],
                         PetscReal ,const Scalar (&U)[nen][dof],
                         Scalar (&F)[nen][dof],
                         void *ctx)
{
  const Params *params = (Params *)ctx;

  using namespace PetIGA;
  typedef  Space<dim,nen> Space;
  typename Space::Shape0 N0 = Space::GetShape0(q);
  typename Space::Shape1 N1 = Space::GetShape1(q);
  typename Space::Shape2 N2 = Space::GetShape2(q);

  Scalar c_t,c;
  Point::FormValue(q,V,c_t);
  Point::FormValue(q,U,c);
  Scalar grad_c[dim],del2_c;
  Point::FormGrad(q,U,grad_c);
  Point::FormDel2<dim>(q,U,del2_c);

  Scalar M,dM;
  Mobility(params,c,M,dM);
  Scalar mu,dmu;
  ChemicalPotential(params,c,mu,dmu);
  Scalar t1 = M*dmu + dM*del2_c;

  for (int a=0; a<nen; a++)
    F[a][0] = N0[a] * c_t + dot(N1[a],grad_c) * t1 + del2(N2[a]) * M * del2_c;

  return 0;
}

template <int dim, int nen, int dof, typename Scalar>
static inline
PetscErrorCode IJacobian(IGAPoint  q,
                         PetscReal s, const Scalar (&V)[nen][dof],
                         PetscReal  , const Scalar (&U)[nen][dof],
                         Scalar (&J)[nen][dof][nen][dof],
                         void *ctx)
{
  const Params *user = (Params *)ctx;

  using namespace PetIGA;
  typedef  Space<dim,nen> Space;
  typename Space::Shape0 N0 = Space::GetShape0(q);
  typename Space::Shape1 N1 = Space::GetShape1(q);
  typename Space::Shape2 N2 = Space::GetShape2(q);

  Scalar c_t,c;
  Point::FormValue(q,V,c_t);
  Point::FormValue(q,U,c);
  Scalar grad_c[dim],del2_c;
  Point::FormGrad(q,U,grad_c);
  Point::FormDel2<dim>(q,U,del2_c);

  Scalar M,dM,d2M;
  Mobility(user,c,M,dM,d2M);
  Scalar mu,dmu,d2mu;
  ChemicalPotential(user,c,mu,dmu,d2mu);
  Scalar t1 = M*dmu + dM*del2_c;
  Scalar t2 = (dM*dmu+M*d2mu+d2M*del2_c);

  for (int a=0; a<nen; a++) {
    PetscReal del2_Na = del2(N2[a]);
    Scalar grad_Na_grad_c = dot(N1[a],grad_c);
    for (int b=0; b<nen; b++) {
      PetscReal del2_Nb = del2(N2[b]);
      Scalar t3 = t2*N0[b] + dM*del2_Nb;
      J[a][0][b][0]  = s * N0[a] * N0[b];
      J[a][0][b][0] += dot(N1[a],N1[b]) * t1;
      J[a][0][b][0] += grad_Na_grad_c   * t3;
      J[a][0][b][0] += del2_Na * (dM*del2_c*N0[b] + M*del2_Nb);
    }
  }

  return 0;
}


#define DOF 1
#include "petiga/cxx/ifunction.hpp"
#include "petiga/cxx/ijacobian.hpp"
