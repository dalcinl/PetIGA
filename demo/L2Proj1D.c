#include "petiga.h"

PetscScalar Line(PetscReal x)
{
  return x;
}

PetscScalar Parabola(PetscReal x)
{
  return x*x;
}

PetscScalar Poly3(PetscReal x)
{
  return x*(x-1)*(x+1);
}

PetscScalar Poly4(PetscReal x)
{
  PetscReal a = 1/3.0;
  return (x-1)*(x-a)*(x+a)*(x+1);
}


typedef struct {
  PetscScalar (*Function)(PetscReal x);
} AppCtx;


#undef  __FUNCT__
#define __FUNCT__ "System"
PetscErrorCode System(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;
  PetscReal x = p->point[0];
  PetscReal *N = p->shape[0];
  
  PetscInt a,b,nen=p->nen;
  for (a=0; a<nen; a++) {
    PetscReal Na = N[a];
    for (b=0; b<nen; b++) {
      PetscReal Nb = N[b];
      K[a*nen+b] = Na * Nb;
    }
    F[a] = Na * user->Function(x);
  }
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[]) {

  PetscErrorCode  ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  AppCtx user;

  PetscInt choice=2;
  const char *choicelist[] = {"line", "parabola", "poly3", "poly4", 0};
  PetscBool t=PETSC_FALSE;
  PetscInt N=16, p=2, C=PETSC_DECIDE;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","Projection1D Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-periodic", "periodic",__FILE__,t,&t,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-N", "number of elements (along one dimension)",__FILE__,N,&N,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-p", "polynomial order",__FILE__,p,&p,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-C", "global continuity order",__FILE__,C,&C,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEList("-function","1D function",__FILE__,choicelist,4,choicelist[choice],&choice,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (C == PETSC_DECIDE) C = p-1;
  switch (choice) {
  case 0: user.Function = Line;     break;
  case 1: user.Function = Parabola; break;
  case 2: user.Function = Poly3;    break;
  case 3: user.Function = Poly4;    break;
  }

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDim(iga,1);CHKERRQ(ierr);
  ierr = IGASetDof(iga,1);CHKERRQ(ierr);
  IGAAxis axis;
  ierr = IGAGetAxis(iga,0,&axis);CHKERRQ(ierr);
  ierr = IGAAxisSetPeriodic(axis,t);CHKERRQ(ierr);
  ierr = IGAAxisSetDegree(axis,p);CHKERRQ(ierr);
  ierr = IGAAxisInitUniform(axis,N,-1.0,1.0,C);CHKERRQ(ierr);

  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);

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

  ierr = VecView(x,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
