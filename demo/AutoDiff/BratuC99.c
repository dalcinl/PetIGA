#include <petiga.h>

typedef struct {
  PetscReal lambda;
} Params;

static inline
PetscScalar dots(PetscInt dim,const PetscReal a[],const PetscScalar b[])
{
  PetscInt i;
  PetscScalar s = 0;
  for (i=0; i<dim; i++) s += a[i]*b[i];
  return s;
}

static inline
PetscReal dotr(PetscInt dim,const PetscReal a[],const PetscReal b[])
{
  PetscInt i;
  PetscReal s = 0;
  for (i=0; i<dim; i++) s += a[i]*b[i];
  return s;
}

extern
PetscErrorCode FunctionC99(IGAPoint q,const PetscScalar U[],PetscScalar R[],void *ctx)
{
  const PetscInt dim = q->dim;
  const PetscInt nen = q->nen;
  typedef const PetscReal (*Shape0);
  typedef const PetscReal (*Shape1)[dim];
  const Shape0 N0 = (Shape0) q->shape[0];
  const Shape1 N1 = (Shape1) q->shape[1];
  const PetscReal lambda = ((Params*)ctx)->lambda;

  PetscScalar u,grad_u[dim];
  IGAPointFormValue(q,U,&u);
  IGAPointFormGrad (q,U,grad_u);

  PetscScalar lambda_exp_u = lambda*exp(u);

  PetscInt a;
  for (a=0; a<nen; a++)
    R[a] = dots(dim,N1[a],grad_u) - lambda_exp_u*N0[a];

  return 0;
}

extern
PetscErrorCode JacobianC99(IGAPoint q,const PetscScalar U[],PetscScalar J[],void *ctx)
{
  const PetscInt dim = q->dim;
  const PetscInt nen = q->nen;
  typedef const PetscReal (*Shape0);
  typedef const PetscReal (*Shape1)[dim];
  const Shape0 N0 = (Shape0) q->shape[0];
  const Shape1 N1 = (Shape1) q->shape[1];
  const PetscReal lambda = ((Params*)ctx)->lambda;

  PetscScalar u;
  IGAPointFormValue(q,U,&u);

  PetscScalar lambda_exp_u = lambda*exp(u);

  PetscInt a,b;
  for (a=0; a<nen; a++)
    for (b=0; b<nen; b++)
      J[a*nen+b] = dotr(dim,N1[a],N1[b]) - lambda_exp_u*N0[a]*N0[b];

  return 0;
}
