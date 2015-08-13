#include "petiga/petiga.hpp"
#include "CahnHilliard.hpp"

template <int dim, int nen, int dof, typename Scalar>
static inline
PetscErrorCode IFunction(IGAPoint  q,
                         PetscReal,const Scalar (&V)[nen][dof],
                         PetscReal,const Scalar (&U)[nen][dof],
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

#define TEST 0

#if TEST
  {

    Scalar c[1];
    Point::FormValue<0>(q,U,c);

    Scalar grad_c[1][dim];
    Point::FormGrad<0>(q,U,grad_c);

    Scalar hess_c[1][dim][dim];
    Point::FormHess<0>(q,U,hess_c);

    Scalar der3_c[1][dim][dim][dim];
    Point::FormDer3<0>(q,U,der3_c);

    Scalar del2_c[1];
    Point::FormDel2<0,dim>(q,U,del2_c);
  }
#endif

#if TEST
  {
    Scalar c[1];
    Point::FormValue(q,U,c);

    Scalar grad_c[1][dim];
    Point::FormGrad(q,U,grad_c);

    Scalar hess_c[1][dim][dim];
    Point::FormHess(q,U,hess_c);

    Scalar der3_c[1][dim][dim][dim];
    Point::FormDer3(q,U,der3_c);

    Scalar del2_c[1];
    Point::FormDel2<dim>(q,U,del2_c);
  }
#endif

#if TEST
  {
    Scalar c;
    Point::FormValue(q,U,c);

    Scalar grad_c[dim];
    Point::FormGrad(q,U,grad_c);

    Scalar hess_c[dim][dim];
    Point::FormHess(q,U,hess_c);

    Scalar der3_c[dim][dim][dim];
    Point::FormDer3(q,U,der3_c);

    Scalar del2_c;
    Point::FormDel2<dim>(q,U,del2_c);
  }
#endif

#if TEST
  {
    Scalar c;
    Point::FormValue<0>(q,U,c);

    Scalar grad_c[dim];
    Point::FormGrad<0>(q,U,grad_c);

    Scalar hess_c[dim][dim];
    Point::FormHess<0>(q,U,hess_c);

    Scalar der3_c[dim][dim][dim];
    Point::FormDer3<0>(q,U,der3_c);

    Scalar del2_c;
    Point::FormDel2<0,dim>(q,U,del2_c);
  }
#endif

  return 0;
}

#include "petiga/fad/ijacobian.hpp"


#define DOF 1
#define IFunctionCXX IFunctionFAD
#define IJacobianCXX IJacobianFAD
#include "petiga/cxx/ifunction.hpp"
#include "petiga/cxx/ijacobian.hpp"
