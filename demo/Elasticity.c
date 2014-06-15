/*
  This code solves the 3D elasticity equations given the Lame
  constants and subject to Dirichlet and Neumann boundary conditions.

  keywords: steady, vector, linear
 */
#include "petiga.h"

#if PETSC_VERSION_LT(3,5,0)
#define KSPSetOperators(ksp,A,B) KSPSetOperators(ksp,A,B,SAME_NONZERO_PATTERN)
#endif

typedef struct {
  PetscReal mu;
  PetscReal lambda;
} AppCtx;

PETSC_STATIC_INLINE
PetscReal DOT(PetscInt dim,const PetscReal a[],const PetscReal b[])
{
  PetscInt i; PetscReal s = 0.0;
  for (i=0; i<dim; i++) s += a[i]*b[i];
  return s;
}

#undef  __FUNCT__
#define __FUNCT__ "System"
PetscErrorCode System(IGAPoint p,PetscScalar *KK,PetscScalar *FF,void *ctx)
{
  AppCtx    *user  = (AppCtx *)ctx;
  PetscReal mu     = user->mu;
  PetscReal lambda = user->lambda;

  PetscInt a,b,nen = p->nen;
  PetscInt i,j,dim = p->dim;

  PetscReal   (*B)                = (typeof(B)) p->shape[0];
  PetscReal   (*D)[dim]           = (typeof(D)) p->shape[1];
  PetscScalar (*K)[dim][nen][dim] = (typeof(K)) KK;
  PetscScalar (*F)[dim]           = (typeof(F)) FF;

  for (a=0; a<nen; a++) {
    for (b=0; b<nen; b++) {
      PetscReal Kabii = DOT(dim,D[a],D[b]);
      for (i=0; i<dim; i++)
        K[a][i][b][i] += mu * Kabii;
      for (i=0; i<dim; i++)
        for (j=0; j<dim; j++)
          K[a][i][b][j] += lambda * D[a][i]*D[b][j] + mu * D[a][j]*D[b][i];
    }
  }

  for (a=0; a<nen; a++)
    for (i=0; i<dim; i++)
      F[a][i] = B[a] * 0.0;

  return 0;
}

#undef  __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[]) {

  PetscInt       i,dim;
  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  // Define problem parameters
  AppCtx user;
  user.mu     = 1.0;
  user.lambda = 1.0;
  PetscInt  nld = 3;
  PetscReal load[3] = {1.0, 0.0, 0.0};

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","Elasticity Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsReal     ("-lambda","Lame's first parameter", __FILE__,user.lambda,&user.lambda,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal     ("-mu",    "Lame's second parameter",__FILE__,user.mu,    &user.mu,    NULL);CHKERRQ(ierr);
  ierr = PetscOptionsRealArray("-load",  "Boundary load",          __FILE__,load,&nld,               NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = IGAOptionsAlias("-d",  "2", "-iga_dim");
  ierr = IGAOptionsAlias("-N", "32", "-iga_elements");
  ierr = IGAOptionsAlias("-p", NULL, "-iga_degree");
  ierr = IGAOptionsAlias("-k", NULL, "-iga_continuity");
  ierr = IGAOptionsAlias("-q", NULL, "-iga_quadrature");

  // Setup discretization
  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
  ierr = IGASetDof(iga,dim);CHKERRQ(ierr);
  ierr = IGASetOrder(iga,1);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  // Set boundary conditions
  for (i=0; i<dim; i++) {ierr = IGASetBoundaryValue(iga,0,0,i,0.0);CHKERRQ(ierr);}     // Dirichlet
  for (i=0; i<dim; i++) {ierr = IGASetBoundaryLoad (iga,0,1,i,load[i]);CHKERRQ(ierr);} // Neumann

  // Create linear system
  Mat A;
  Vec x,b;
  ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&b);CHKERRQ(ierr);
  ierr = IGASetFormSystem(iga,System,&user);CHKERRQ(ierr);
  ierr = IGAComputeSystem(iga,A,b);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_SPD,PETSC_TRUE);CHKERRQ(ierr);

  // Solve linear system
  KSP ksp;
  ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  // Save geometry and solution vector
  PetscBool save = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,"-save",&save,NULL);CHKERRQ(ierr);
  if (save) {
    ierr = IGAWrite(iga,"Elasticity-geometry.dat");CHKERRQ(ierr);
    ierr = IGAWriteVec(iga,x,"Elasticity-solution.dat");CHKERRQ(ierr);
  }

  // Draw solution vector
  PetscBool draw = PETSC_FALSE;
  PetscReal pause = 0.0;
  ierr = PetscOptionsGetBool(NULL,"-draw",&draw,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,"-pause",&pause,NULL);CHKERRQ(ierr);
  if (draw && dim <= 2) {
    ierr = VecView(x,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
    ierr = PetscSleep(pause);CHKERRQ(ierr);
  }

  // Cleanup
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}