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

PetscScalar Hill(PetscReal x, PetscReal y)
{
  return   (x-1)*(x-1)*(x+1)*(x+1)
    /**/ * (y-1)*(y-1)*(y+1)*(y+1);
}

PetscScalar Sine(PetscReal x, PetscReal y)
{
  return   sin(M_PI*x)
    /**/ * sin(M_PI*y);
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

#undef  __FUNCT__
#define __FUNCT__ "Error"
PetscErrorCode Error(IGAPoint p,const PetscScalar *U,PetscInt n,PetscScalar *S,void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;
  PetscReal x = p->point[0];
  PetscReal y = p->point[1];
  PetscScalar f = user->Function(x,y);

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

  PetscBool flg;
  PetscBool w[2] = {PETSC_FALSE,PETSC_FALSE}; PetscInt nw = 2;
  PetscInt  N[2] = {16,16}, nN = 2;
  PetscInt  p[2] = { 2, 2}, np = 2;
  PetscInt  C[2] = {-1,-1}, nC = 2;
  PetscInt  choice=0;
  const char *choicelist[] = {"paraboloid", "peaks", "hill", "sine", 0};
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","L2 Projection 2D Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsBoolArray("-periodic", "periodicity",    __FILE__,w,&nw,&flg);CHKERRQ(ierr);
  if (flg && nw==0) w[nw++] = PETSC_TRUE;
  ierr = PetscOptionsIntArray ("-N","number of elements",__FILE__,N,&nN,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray ("-p","polynomial order",  __FILE__,p,&np,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray ("-C","continuity order",  __FILE__,C,&nC,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEList    ("-function","2D function",__FILE__,choicelist,4,choicelist[choice],&choice,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (nw == 1) w[1] = w[0];
  if (nN == 1) N[1] = N[0];
  if (np == 1) p[1] = p[0];
  if (nC == 1) C[1] = C[0];
  if (C[0] == -1) C[0] = p[0]-1;
  if (C[1] == -1) C[1] = p[1]-1;
  switch (choice) {
  case 0: user.Function = Paraboloid; break;
  case 1: user.Function = Peaks;      break;
  case 2: user.Function = Hill;       break;
  case 3: user.Function = Sine;       break;
  }

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDim(iga,2);CHKERRQ(ierr);
  ierr = IGASetDof(iga,1);CHKERRQ(ierr);
  PetscInt i;
  for (i=0; i<2; i++) {
    IGAAxis axis;
    ierr = IGAGetAxis(iga,i,&axis);CHKERRQ(ierr);
    ierr = IGAAxisSetPeriodic(axis,w[i]);CHKERRQ(ierr);
    ierr = IGAAxisSetDegree(axis,p[i]);CHKERRQ(ierr);
    ierr = IGAAxisInitUniform(axis,N[i],-1.0,1.0,C[i]);CHKERRQ(ierr);
  }
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
