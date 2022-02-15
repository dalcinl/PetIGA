/*
  This code solves the Laplace problem where the boundary conditions
  can be changed from Neumann to Dirichlet via the commandline. While
  its primary use is in regression tests for PetIGA, it also
  demonstrates how boundary integrals may be performed to enforce
  things like Neumann conditions.

  keywords: steady, scalar, linear, testing, dimension independent,
  boundary integrals
 */
#include "petiga.h"

typedef struct {
  PetscInt axis;
  PetscInt side;
} AppCtx;

static inline
PetscReal DOT(PetscInt dim,const PetscReal a[],const PetscReal b[])
{
  PetscInt i; PetscReal s = 0.0;
  for (i=0; i<dim; i++) s += a[i]*b[i];
  return s;
}

PetscErrorCode Laplace(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  PetscInt dim = p->dim;
  PetscInt nen = p->nen;
  const PetscReal (*N1)[dim] = (typeof(N1)) p->shape[1];
  PetscInt a,b;
  for (a=0; a<nen; a++) {
    for (b=0; b<nen; b++)
      K[a*nen+b] = DOT(dim,N1[a],N1[b]);
    F[a] = 0.0;
  }
  return 0;
}

PetscErrorCode Neumann(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  PetscInt nen = p->nen;
  const PetscReal *N0 = (typeof(N0)) p->shape[0];
  PetscInt a;
  for (a=0; a<nen; a++)
    F[a] = N0[a] * 1.0;
  return 0;
}

PetscErrorCode System(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  if (!p->atboundary)
    return Laplace(p,K,F,ctx);
  else
    return Neumann(p,K,F,ctx);
}


static inline
PetscReal DEL2(PetscInt dim,const PetscReal a[dim][dim])
{
  PetscInt i; PetscReal s = 0.0;
  for (i=0; i<dim; i++) s += a[i][i];
  return s;
}

PetscErrorCode LaplaceCollocation(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  PetscInt nen = p->nen;
  PetscInt dim = p->dim;
  const PetscReal (*N2)[dim][dim] = (typeof(N2)) p->shape[2];

  PetscInt a;
  for (a=0; a<nen; a++)
    K[a] = -DEL2(dim,N2[a]);
  F[0] = 0.0;

  return 0;
}

PetscErrorCode DirichletCollocation(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  PetscInt nen = p->nen;
  const PetscReal *N0 = (typeof(N0)) p->shape[0];
  PetscInt a;
  for (a=0; a<nen; a++)
    K[a] = N0[a];
  F[0] = 1.0;
  return 0;
}

PetscErrorCode NeumannCollocation(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  PetscInt nen = p->nen;
  PetscInt dim = p->dim;
  const PetscReal (*N1)[dim] = (typeof(N1)) p->shape[1];
  const PetscReal *normal    = p->normal;
  PetscInt a;
  for (a=0; a<nen; a++)
    K[a] = DOT(dim,N1[a],normal);
  F[0] = 1.0;
  return 0;
}
PetscErrorCode Neumann0Collocation(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{(void)NeumannCollocation(p,K,F,ctx); F[0] = 0.0; return 0;}


PetscErrorCode SystemCollocation(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  if (!p->atboundary) return LaplaceCollocation(p,K,F,ctx);

  AppCtx *user = (AppCtx *)ctx;
  PetscInt axis,side;
  IGAPointAtBoundary(p,&axis,&side);
  if (axis == user->axis) {
    if (side == user->side)
      return NeumannCollocation(p,K,F,ctx);
    else
      return DirichletCollocation(p,K,F,ctx);
  } else
    return Neumann0Collocation(p,K,F,ctx);
}

PetscErrorCode Exact(IGAPoint p,PetscInt order,PetscScalar value[],void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;
  PetscReal x;
  if (user->side == 0)
    x = 1 - p->point[user->axis] + 1;
  else
    x = p->point[user->axis] + 1;
  value[0] = x;
  return 0;
}

int main(int argc, char *argv[]) {

  PetscErrorCode  ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  AppCtx user;
  user.axis = 0;
  user.side = 1;

  PetscBool print_error = PETSC_FALSE;
  PetscBool check_error = PETSC_FALSE;
  PetscBool draw = PETSC_FALSE;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","BoundaryIntegral Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsInt ("-axis","Neuman BC direction",__FILE__,user.axis,&user.axis,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt ("-side","Neuman BC side",     __FILE__,user.side,&user.side,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-print_error","Prints the error of the solution",__FILE__,print_error,&print_error,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-check_error","Checks the error of the solution",__FILE__,check_error,&check_error,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-draw","If dim <= 2, then draw the solution to the screen",__FILE__,draw,&draw,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDof(iga,1);CHKERRQ(ierr);
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  if (iga->dim < 1) {ierr = IGASetDim(iga,2);CHKERRQ(ierr);}

  PetscInt dim;
  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);

  IGAForm form;
  ierr = IGAGetForm(iga,&form);CHKERRQ(ierr);
  if (!iga->collocation) {
    PetscInt d = !user.side;
    PetscInt n = !d;
    ierr = IGAFormSetSystem(form,System,&user);CHKERRQ(ierr);
    ierr = IGAFormSetBoundaryValue(form,user.axis,d,0,1.0);CHKERRQ(ierr);
    ierr = IGAFormSetBoundaryForm (form,user.axis,n,PETSC_TRUE);CHKERRQ(ierr);
  } else {
    PetscInt i,s;
    ierr = IGAFormSetSystem(form,SystemCollocation,&user);CHKERRQ(ierr);
    for (i=0; i<dim; i++)
      for (s=0; s<2; s++)
        {ierr = IGAFormSetBoundaryForm(form,i,s,PETSC_TRUE);CHKERRQ(ierr);}
  }
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  Mat A;
  Vec x,b;
  ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&b);CHKERRQ(ierr);
  ierr = IGAComputeSystem(iga,A,b);CHKERRQ(ierr);

  KSP ksp;
  ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  PetscReal error;
  ierr = IGAComputeErrorNorm(iga,0,x,Exact,&error,&user);CHKERRQ(ierr);

  if (print_error) {ierr = PetscPrintf(PETSC_COMM_WORLD,"Error = %g\n",(double)error);CHKERRQ(ierr);}
  if (check_error) {if (error>1e-3) SETERRQ(PETSC_COMM_WORLD,1,"Error=%g\n",(double)error);}
  if (draw&&dim<3) {ierr = IGADrawVec(iga,x,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
