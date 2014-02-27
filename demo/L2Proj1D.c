#include "petiga.h"

#if PETSC_VERSION_LT(3,5,0)
#define KSPSetOperators(ksp,A,B) KSPSetOperators(ksp,A,B,SAME_NONZERO_PATTERN)
#endif

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

PetscScalar Hill(PetscReal x)
{
  return (x-1)*(x-1)*(x+1)*(x+1);
}

PetscScalar Sine(PetscReal x)
{
  return sin(M_PI*x);
}

PetscScalar Step(PetscReal x)
{
  return (x<0.0)?-1.0:+1.0;
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
  PetscScalar f = user->Function(x);

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

#undef  __FUNCT__
#define __FUNCT__ "Error"
PetscErrorCode Error(IGAPoint p,const PetscScalar *U,PetscInt n,PetscScalar *S,void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;
  PetscReal x = p->point[0];
  PetscScalar f = user->Function(x);

  PetscScalar u;
  IGAPointFormValue(p,U,&u);

  PetscReal e = PetscAbsScalar(u - f);
  S[0] = e*e;

  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[]) {

  PetscErrorCode  ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  AppCtx user;

  PetscBool w=PETSC_FALSE;
  PetscInt  N=16;
  PetscInt  p=2;
  PetscInt  C=PETSC_DECIDE;
  PetscInt  choice=2;
  const char *choicelist[] = {"line", "parabola", "poly3", "poly4", "hill", "sine", "step", 0};
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","L2 Projection 1D Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsBool ("-periodic", "periodicity",__FILE__,w,&w,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt  ("-N", "number of elements",__FILE__,N,&N,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt  ("-p", "polynomial order",  __FILE__,p,&p,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt  ("-C", "continuity order",  __FILE__,C,&C,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEList("-function","1D function", __FILE__,choicelist,7,choicelist[choice],&choice,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (C == PETSC_DECIDE) C = p-1;
  switch (choice) {
  case 0: user.Function = Line;     break;
  case 1: user.Function = Parabola; break;
  case 2: user.Function = Poly3;    break;
  case 3: user.Function = Poly4;    break;
  case 4: user.Function = Hill;     break;
  case 5: user.Function = Sine;     break;
  case 6: user.Function = Step;     break;
  }

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDim(iga,1);CHKERRQ(ierr);
  ierr = IGASetDof(iga,1);CHKERRQ(ierr);
  IGAAxis axis;
  ierr = IGAGetAxis(iga,0,&axis);CHKERRQ(ierr);
  ierr = IGAAxisSetPeriodic(axis,w);CHKERRQ(ierr);
  ierr = IGAAxisSetDegree(axis,p);CHKERRQ(ierr);
  ierr = IGAAxisInitUniform(axis,N,-1.0,1.0,C);CHKERRQ(ierr);

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
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  PetscScalar error = 0;
  ierr = IGAComputeScalar(iga,x,1,&error,Error,&user);CHKERRQ(ierr);
  error = PetscSqrtReal(PetscRealPart(error));
  PetscBool print_error = PETSC_FALSE;
  ierr = PetscOptionsGetBool(0,"-error",&print_error,0);CHKERRQ(ierr);
  if (print_error) {ierr = PetscPrintf(PETSC_COMM_WORLD,"L2 error = %G\n",error);CHKERRQ(ierr);}

  PetscBool draw = PETSC_FALSE;
  ierr = PetscOptionsGetBool(0,"-draw",&draw,0);CHKERRQ(ierr);
  if (draw) {ierr = VecView(x,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
