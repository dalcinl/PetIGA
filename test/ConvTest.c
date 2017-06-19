#include "petiga.h"

typedef struct {
  PetscReal c; /* reaction coefficient */
  PetscReal k; /* diffusion coefficient */
} AppCtx;

PETSC_STATIC_INLINE
void Solution(PetscInt dim,const PetscReal x[],PetscReal *value,PetscReal grad[])
{
  PetscInt  i,j;
  PetscReal omega = PETSC_PI;
  if (value) {
    value[0] = 1;
    for (i=0; i<dim; i++) value[0] *= PetscSinReal(omega*x[i]);
  }
  if (grad) {
    for (i=0; i<dim; i++) {
      grad[i] = 1;
      for (j=0; j<dim; j++) {
        if (i == j)
          grad[i] *= omega*PetscCosReal(omega*x[j]);
        else
          grad[i] *= PetscSinReal(omega*x[j]);
      }
    }
  }
}

PETSC_STATIC_INLINE
PetscReal Forcing(PetscInt dim,const PetscReal x[],const AppCtx *app)
{
  PetscInt i;
  const PetscReal c = app->c;
  const PetscReal k = app->k;
  const PetscReal omega = PETSC_PI;
  PetscReal f = c + k*dim*omega*omega;
  for (i=0; i<dim; i++) f *= PetscSinReal(omega*x[i]);
  return f;
}

PETSC_STATIC_INLINE
PetscReal DOT(PetscInt dim,const PetscReal a[dim],const PetscReal b[dim])
{
  PetscInt i; PetscReal s = 0.0;
  for (i=0; i<dim; i++) s += a[i]*b[i];
  return s;
}

PetscErrorCode Galerkin(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  const AppCtx *app = (AppCtx*)ctx;
  const PetscInt dim = p->dim;
  const PetscInt nen = p->nen;

  const PetscReal c = app->c;
  const PetscReal k = app->k;
  const PetscReal f = Forcing(dim,p->point,app);

  const PetscReal *N0        = (typeof(N0)) p->basis[0];
  const PetscReal (*N1)[dim] = (typeof(N1)) p->basis[1];

  PetscInt a,b;
  for (a=0; a<nen; a++) {
    for (b=0; b<nen; b++)
      K[a*nen+b] = c*N0[a]*N0[b] + k*DOT(dim,N1[a],N1[b]);
    F[a] = N0[a]*f;
  }
  return 0;
}

PETSC_STATIC_INLINE
PetscReal TRACE(PetscInt dim,const PetscReal a[dim][dim])
{
  PetscInt i; PetscReal s = 0.0;
  for (i=0; i<dim; i++) s += a[i][i];
  return s;
}

PetscErrorCode Collocation(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{

  const AppCtx  *app = (AppCtx*)ctx;
  const PetscInt dim = p->dim;
  const PetscInt nen = p->nen;

  const PetscReal c = app->c;
  const PetscReal k = app->k;
  const PetscReal f = Forcing(dim,p->point,app);

  const PetscReal *N0             = (typeof(N0)) p->basis[0];
  const PetscReal (*N2)[dim][dim] = (typeof(N2)) p->basis[2];

  PetscInt a;
  for (a=0; a<nen; a++)
    K[a] = c*N0[a] - k*TRACE(dim,N2[a]);
  F[0] = f;

  return 0;
}

PetscErrorCode Exact(IGAPoint p,PetscInt order,PetscScalar value[],void *ctx)
{
  switch (order) {
  case 0: Solution(p->dim,p->point,value,NULL); break;
  case 1: Solution(p->dim,p->point,NULL,value); break;
  }
  return 0;
}

int main(int argc, char *argv[])
{
  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  ierr = IGAOptionsAlias("-d",  "2", "-iga_dim");
  ierr = IGAOptionsAlias("-n",  "8", "-iga_elements");
  ierr = IGAOptionsAlias("-p", NULL, "-iga_degree");
  ierr = IGAOptionsAlias("-k", NULL, "-iga_continuity");
  ierr = IGAOptionsAlias("-q", NULL, "-iga_rule_size");
  ierr = IGAOptionsAlias("-r", NULL, "-iga_rule_type");

  /* Application options */

  AppCtx app;
  app.c = 1;
  app.k = 1;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","ConvTest Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-reaction", "Reaction  coefficient",__FILE__,app.c,&app.c,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-diffusion","Diffusion coefficient",__FILE__,app.k,&app.k,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* Initialize the discretization */

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDof(iga,1);CHKERRQ(ierr);
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  if (!iga->collocation) {
    ierr = IGASetOrder(iga,1);CHKERRQ(ierr);
  } else {
    ierr = IGASetOrder(iga,2);CHKERRQ(ierr);
  }

  /* Set boundary conditions */

  PetscInt dim,i;
  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
  if (app.k > 0) {
    for (i=0; i<dim; i++) {
      ierr = IGASetBoundaryValue(iga,i,0,0,0.0);CHKERRQ(ierr);
      ierr = IGASetBoundaryValue(iga,i,1,0,0.0);CHKERRQ(ierr);
    }
  }

  /* Assemble */

  Mat A; Vec x,b;
  ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&b);CHKERRQ(ierr);
  if (!iga->collocation) {
    ierr = IGASetFormSystem(iga,Galerkin,&app);CHKERRQ(ierr);
    ierr = IGAComputeSystem(iga,A,b);CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_SPD,PETSC_TRUE);CHKERRQ(ierr);
  } else {
    ierr = IGASetFormSystem(iga,Collocation,&app);CHKERRQ(ierr);
    ierr = IGAComputeSystem(iga,A,b);CHKERRQ(ierr);
  }

  /* Solve */

  KSP ksp; PC pc;
  ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
  #ifdef PETSC_HAVE_MUMPS
  ierr = PCFactorSetMatSolverPackage(pc,MATSOLVERMUMPS);CHKERRQ(ierr);
  #endif
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  /* Compute L2 and H1 errors */

  if (iga->collocation) {ierr = IGAReset(iga);CHKERRQ(ierr);}
  if (iga->collocation) {ierr = IGASetUseCollocation(iga,PETSC_FALSE);CHKERRQ(ierr);}
  for (i=0; i<dim; i++) {ierr = IGASetRuleType(iga,i,IGA_RULE_LEGENDRE);CHKERRQ(ierr);}
  for (i=0; i<dim; i++) {ierr = IGASetRuleSize(iga,i,10);CHKERRQ(ierr);}
  ierr = IGASetUp(iga);CHKERRQ(ierr);
  PetscReal errorL2,seminormH1;
  ierr = IGAComputeErrorNorm(iga,0,x,Exact,&errorL2,NULL);CHKERRQ(ierr);
  ierr = IGAComputeErrorNorm(iga,1,x,Exact,&seminormH1,NULL);CHKERRQ(ierr);
  PetscReal errorH1 = PetscSqrtReal(PetscSqr(errorL2)+PetscSqr(seminormH1));
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Error:  L2 = %.16e  H1 = %.16e\n",(double)errorL2,(double)errorH1);CHKERRQ(ierr);

  /* Cleanup */

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
