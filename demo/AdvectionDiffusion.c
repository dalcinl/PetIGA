/*
  This code solves the advection-diffusion problem where the advection
  is constant and skew to the mesh. It is independent of the spatial
  dimension which must be specified in the commandline options via
  -iga_dim. The strength of advection can be controlled through the
  option -Pe, representing the Peclet number. For example,

  ./AdvectionDiffusion -iga_dim 1 -draw -draw_pause -1 -Pe 10

  keywords: steady, scalar, linear, educational, dimension independent
 */
#include "petiga.h"

typedef struct {
  PetscReal wind[3];
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
PetscErrorCode System(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;
  PetscReal *w = user->wind;

  PetscInt nen = p->nen;
  PetscInt dim = p->dim;
  const PetscReal *N0        = (typeof(N0)) p->shape[0];
  const PetscReal (*N1)[dim] = (typeof(N1)) p->shape[1];

  PetscInt a,b;
  for (a=0; a<nen; a++) {
    for (b=0; b<nen; b++) {
      PetscScalar diffusion = DOT(dim,N1[a],N1[b]);
      PetscScalar advection = N0[a]*DOT(dim,w,N1[b]);
      K[a*nen+b] = diffusion + advection;
    }
    F[a] = 0.0;
  }
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[]) {

  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  AppCtx    user;
  PetscInt  i,dim;
  PetscReal Pe = 1.0;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","AdvectionDiffusion Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-Pe","Peclet number",__FILE__,Pe,&Pe,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  // advection is skew to the mesh
  for (i=0; i<dim; i++) user.wind[i] = Pe*sqrt(dim)/dim;

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDof(iga,1);CHKERRQ(ierr);
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
  IGABoundary bnd;
  for (i=0; i<dim; i++) {
    ierr = IGAGetBoundary(iga,i,0,&bnd);CHKERRQ(ierr);
    ierr = IGABoundarySetValue(bnd,0,1.0);CHKERRQ(ierr);
    ierr = IGAGetBoundary(iga,i,1,&bnd);CHKERRQ(ierr);
    ierr = IGABoundarySetValue(bnd,0,0.0);CHKERRQ(ierr);
  }

  Mat A;
  Vec x,b;
  ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&b);CHKERRQ(ierr);
  ierr = IGASetUserSystem(iga,System,&user);CHKERRQ(ierr);
  ierr = IGAComputeSystem(iga,A,b);CHKERRQ(ierr);

  KSP ksp;
  ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  PetscBool draw = PETSC_FALSE;
  ierr = PetscOptionsGetBool(0,"-draw",&draw,0);CHKERRQ(ierr);
  if (draw&&dim<=2) {ierr = VecView(x,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
