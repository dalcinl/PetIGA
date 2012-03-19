#include "petiga.h"

PetscScalar Paraboloid(PetscReal x, PetscReal y)
{
  return 1.0 - (x*x + y*y);
}

PetscScalar Peaks(PetscReal x, PetscReal y)
{
  PetscReal X = x*3;
  PetscReal Y = y*3;
  return   3 * pow(1-X,2) * exp(-pow(X,2) - pow(Y+1,2))
    /**/ - 10 * (X/5 - pow(X,3) - pow(Y,5)) * exp(-pow(X,2) - pow(Y,2))
    /**/ - 1.0/3 * exp(-pow(X+1,2) - pow(Y,2));
}

typedef struct {
  PetscScalar (*Function)(PetscReal x, PetscReal y);
} AppCtx;


#undef  __FUNCT__
#define __FUNCT__ "System"
PetscErrorCode System(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;
  PetscReal x = p->point[0];
  PetscReal y = p->point[1];
  PetscReal *N = p->shape[0];

  PetscInt a,b,nen=p->nen;
  for (a=0; a<nen; a++) {
    PetscReal Na = N[a];
    for (b=0; b<nen; b++) {
      PetscReal Nb = N[b];
      K[a*nen+b] = Na * Nb;
    }
    F[a] = Na * user->Function(x,y);
  }
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[]) {

  PetscErrorCode  ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  AppCtx user;

  PetscInt choice=0;
  const char *choicelist[] = {"paraboloid", "peaks", 0};
  PetscInt N=16, p=2, C=PETSC_DECIDE;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","Projection2D Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-N", "number of elements (along one dimension)",__FILE__,N,&N,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-p", "polynomial order",__FILE__,p,&p,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-C", "global continuity order",__FILE__,C,&C,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEList("-function","2D function",__FILE__,choicelist,2,choicelist[choice],&choice,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (C == PETSC_DECIDE) C = p-1;
  switch (choice) {
  case 0: user.Function = Paraboloid; break;
  case 1: user.Function = Peaks;      break;
  }

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDim(iga,2);CHKERRQ(ierr);
  ierr = IGASetDof(iga,1);CHKERRQ(ierr);
  IGAAxis axis0;
  ierr = IGAGetAxis(iga,0,&axis0);CHKERRQ(ierr);
  ierr = IGAAxisInitUniform(axis0,p,C,N,-1.0,1.0);CHKERRQ(ierr);
  IGAAxis axis1;
  ierr = IGAGetAxis(iga,1,&axis1);CHKERRQ(ierr);
  ierr = IGAAxisCopy(axis0,axis1);CHKERRQ(ierr);

  ierr = IGASetUp(iga);CHKERRQ(ierr);

  Mat A;
  Vec x,b;
  ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&b);CHKERRQ(ierr);
  ierr = IGAFormSystem(iga,A,b,System,&user);CHKERRQ(ierr);

  KSP ksp;
  ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  ierr = VecView(x,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
