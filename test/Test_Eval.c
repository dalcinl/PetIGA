#include "petiga.h"
#include "time.h"

#define SQ(A) ((A)*(A))

PetscScalar CrossTerm(PetscReal x, PetscReal y)
{
  return x*y;
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
  PetscScalar f = user->Function(x,y);

  PetscReal *N = p->shape[0];
  PetscInt a,b,nen=p->nen;
  for (a=0; a<nen; a++) {
    PetscReal Na = N[a];
    for (b=0; b<nen; b++) {
      PetscReal Nb = N[b];
      K[a*nen+b] = Na * Nb;
    }
    F[a] = Na * f;
  }
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[]) {

  PetscErrorCode  ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  AppCtx user;
  user.Function = CrossTerm;

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDim(iga,2);CHKERRQ(ierr);
  ierr = IGASetDof(iga,1);CHKERRQ(ierr);
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  Mat A;
  Vec x,b;
  ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&b);CHKERRQ(ierr);
  ierr = IGASetFormSystem(iga,System,&user);CHKERRQ(ierr);
  ierr = IGAComputeSystem(iga,A,b);CHKERRQ(ierr);

  KSP ksp;
  ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  PetscReal  pnt[2];
  PetscScalar sol=0,gsol[2],err,gerr;
  srand(time(NULL));
  PetscInt i;
  for(i=0;i<10;i++){
    pnt[0] = rand();
    pnt[1] = rand();
    pnt[0] /= RAND_MAX;
    pnt[1] /= RAND_MAX;
    PetscPrintf(PETSC_COMM_WORLD,"Evaluating solution at x = (%f,%f)\n",pnt[0],pnt[1]);
    ierr = IGAInterpolate(iga,x,pnt,&sol,gsol);
    err = fabs(sol-user.Function(pnt[0],pnt[1]));
    gerr = sqrt(SQ(gsol[0]-pnt[1])+SQ(gsol[1]-pnt[0]));
    PetscPrintf(PETSC_COMM_WORLD,"\t     Solution  =  %e                Error = %e\n",sol,err);
    PetscPrintf(PETSC_COMM_WORLD,"\tgrad(Solution) = [%e %e]  Error = %e\n",gsol[0],gsol[1],gerr);
  }

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
