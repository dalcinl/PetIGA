#include <petiga.h>
#include "CahnHilliard.h"

extern
PetscErrorCode IFunctionC99(IGAPoint  q,
                            PetscReal s,const PetscScalar V[],
                            PetscReal t,const PetscScalar U[],
                            PetscScalar F[],void *ctx)
{
  const Params *user = (Params *)ctx;
  const PetscInt dim = q->dim;
  const PetscInt nen = q->nen;
  typedef const PetscReal (*Shape0);
  typedef const PetscReal (*Shape1)[dim];
  typedef const PetscReal (*Shape2)[dim][dim];
  const Shape0 N0 = (Shape0) q->shape[0];
  const Shape1 N1 = (Shape1) q->shape[1];
  const Shape2 N2 = (Shape2) q->shape[2];

  PetscScalar c_t,c;
  IGAPointFormValue(q,V,&c_t);
  IGAPointFormValue(q,U,&c);
  PetscScalar grad_c[dim],del2_c;
  IGAPointFormGrad(q,U,grad_c);
  IGAPointFormDel2(q,U,&del2_c);

  PetscScalar M,dM;
  Mobility(user,c,&M,&dM,NULL);
  PetscScalar mu,dmu;
  ChemicalPotential(user,c,&mu,&dmu,NULL);
  PetscScalar t1 = M*dmu + dM*del2_c;

  PetscInt a;
  for (a=0; a<nen; a++)
    F[a] = N0[a] * c_t + dots(dim,N1[a],grad_c) * t1 + del2(dim,N2[a]) * M * del2_c;

  return 0;
}

extern
PetscErrorCode IJacobianC99(IGAPoint  q,
                            PetscReal s,const PetscScalar V[],
                            PetscReal t,const PetscScalar U[],
                            PetscScalar J[],void *ctx)
{
  const Params *user = (Params *)ctx;
  const PetscInt dim = q->dim;
  const PetscInt nen = q->nen;
  typedef const PetscReal (*Shape0);
  typedef const PetscReal (*Shape1)[dim];
  typedef const PetscReal (*Shape2)[dim][dim];
  const Shape0 N0 = (Shape0) q->shape[0];
  const Shape1 N1 = (Shape1) q->shape[1];
  const Shape2 N2 = (Shape2) q->shape[2];

  PetscScalar c_t,c;
  IGAPointFormValue(q,V,&c_t);
  IGAPointFormValue(q,U,&c);
  PetscScalar grad_c[dim],del2_c;
  IGAPointFormGrad(q,U,grad_c);
  IGAPointFormDel2(q,U,&del2_c);

  PetscScalar M,dM,d2M;
  Mobility(user,c,&M,&dM,&d2M);
  PetscScalar mu,dmu,d2mu;
  ChemicalPotential(user,c,&mu,&dmu,&d2mu);
  PetscScalar t1 = M*dmu + dM*del2_c;
  PetscScalar t2 = (dM*dmu+M*d2mu+d2M*del2_c);

  PetscInt a,b;
  for (a=0; a<nen; a++) {
    PetscReal   del2_Na = del2(dim,N2[a]);
    PetscScalar grad_Na_grad_c = dots(dim,N1[a],grad_c);
    for (b=0; b<nen; b++) {
      PetscReal del2_Nb = del2(dim,N2[b]);
      PetscScalar t3 = t2*N0[b] + dM*del2_Nb;
      J[a*nen+b]  = s * N0[a] * N0[b];
      J[a*nen+b] += dotr(dim,N1[a],N1[b]) * t1;
      J[a*nen+b] += grad_Na_grad_c * t3;
      J[a*nen+b] += del2_Na * (dM*del2_c*N0[b] + M*del2_Nb);
    }
  }

  return 0;
}
