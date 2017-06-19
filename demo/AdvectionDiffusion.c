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

PetscErrorCode ComputeWind(PetscInt dim,PetscReal Pe,const PetscReal dir[],PetscReal wind[])
{
  PetscInt  i;
  PetscReal norm = 0;
  for (i=0; i<dim; i++) norm += dir[i]*dir[i];
  norm = PetscSqrtReal(norm); if (norm<1e-2) norm = 1.0;
  for (i=0; i<dim; i++) wind[i] = Pe*dir[i]/norm;
  return 0;
}

int main(int argc, char *argv[]) {

  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  PetscInt  i,dim,n=3;
  PetscReal Pe = 1.0;
  PetscReal dir[3] = {1.0,1.0,1.0};
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","AdvectionDiffusion Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-Pe","Peclet number",__FILE__,Pe,&Pe,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsRealArray("-dir","Direction",__FILE__,dir,&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDof(iga,1);CHKERRQ(ierr);
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
  if (dim<1) {ierr = IGASetDim(iga,dim=2);CHKERRQ(ierr);}
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  AppCtx user;
  ierr = ComputeWind(dim,Pe,dir,user.wind);CHKERRQ(ierr);

  PetscReal ul=1.0, ur=0.0;
  for (i=0; i<dim; i++) {
    if (dir[i] == 0.0) continue;
    ierr = IGASetBoundaryValue(iga,i,0,0,ul);CHKERRQ(ierr);
    ierr = IGASetBoundaryValue(iga,i,1,0,ur);CHKERRQ(ierr);
  }

  Mat A;
  Vec x,b;
  ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&b);CHKERRQ(ierr);
  ierr = IGASetFormSystem(iga,System,&user);CHKERRQ(ierr);
  ierr = IGAComputeSystem(iga,A,b);CHKERRQ(ierr);

  KSP ksp;
  ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  PetscBool draw = IGAGetOptBool(NULL,"-draw",PETSC_FALSE);
  if (draw&&dim<3) {ierr = IGADrawVec(iga,x,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
