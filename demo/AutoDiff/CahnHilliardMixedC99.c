#include <petiga.h>
#include "CahnHilliard.h"

extern
PetscErrorCode IFunctionC99(IGAPoint  q,
                            PetscReal _,const PetscScalar V[],
                            PetscReal t,const PetscScalar U[],
                            PetscScalar FF[],void *ctx)
{
  const Params *params = (Params *)ctx;
  const PetscInt dim = q->dim;
  const PetscInt nen = q->nen;
  typedef const PetscReal (*Shape0);
  typedef const PetscReal (*Shape1)[dim];
  const Shape0 N0 = (Shape0) q->shape[0];
  const Shape1 N1 = (Shape1) q->shape[1];

  typedef PetscScalar (*ArrayF)[2];
  ArrayF F = (ArrayF) FF;

  PetscScalar u_t[2],u[2],grad_u[2][dim];
  IGAPointFormValue(q,V,&u_t[0]);
  IGAPointFormValue(q,U,&u[0]);
  IGAPointFormGrad (q,U,&grad_u[0][0]);
  PetscScalar c_t = u_t[0];
  PetscScalar c = u[0], *grad_c = grad_u[0];
  PetscScalar s = u[1], *grad_s = grad_u[1];

  PetscScalar M;
  Mobility(params,c,&M,NULL,NULL);
  PetscScalar mu;
  ChemicalPotential(params,c,&mu,NULL,NULL);

  PetscInt a;
  for (a=0; a<nen; a++) {
    F[a][0] = N0[a] *  c_t   + dots(dim,N1[a],grad_s) * M;
    F[a][1] = N0[a] * (s-mu) - dots(dim,N1[a],grad_c);
  }

  return 0;
}

extern
PetscErrorCode IJacobianC99(IGAPoint  q,
                            PetscReal shift,const PetscScalar V[],
                            PetscReal t,const PetscScalar U[],
                            PetscScalar JJ[],void *ctx)
{
  const Params *params = (Params *)ctx;
  const PetscInt dim = q->dim;
  const PetscInt nen = q->nen;
  typedef const PetscReal (*Shape0);
  typedef const PetscReal (*Shape1)[dim];
  const Shape0 N0 = (Shape0) q->shape[0];
  const Shape1 N1 = (Shape1) q->shape[1];

  typedef PetscScalar (*ArrayJ)[2][nen][2];
  ArrayJ J = (ArrayJ) JJ;

  PetscScalar u[2],grad_u[2][dim];
  IGAPointFormValue(q,U,&u[0]);
  IGAPointFormGrad (q,U,&grad_u[0][0]);
  PetscScalar c = u[0], *grad_s = grad_u[1];

  PetscScalar M,dM;
  Mobility(params,c,&M,&dM,NULL);
  PetscScalar mu,dmu;
  ChemicalPotential(params,c,&mu,&dmu,NULL);

  PetscInt a,b;
  for (a=0; a<nen; a++) {
    PetscScalar dNa_dot_gs_dM = dots(dim,N1[a],grad_s) * dM;
    for (b=0; b<nen; b++) {
      PetscScalar gNa_dot_gNb = dotr(dim,N1[a],N1[b]);
      J[a][0][b][0] += shift * N0[a] * N0[b] + dNa_dot_gs_dM * N0[b];
      J[a][0][b][1] += M * gNa_dot_gNb;
      J[a][1][b][0] += - dmu * N0[a] * N0[b] - gNa_dot_gNb;
      J[a][1][b][1] += N0[a] * N0[b];
    }
  }

  return 0;
}
