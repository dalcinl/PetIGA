#include "petiga.h"

#if PETSC_VERSION_LT(3,5,0)
#define KSPSetOperators(ksp,A,B) KSPSetOperators(ksp,A,B,SAME_NONZERO_PATTERN)
#endif

PetscScalar Function(PetscReal x, PetscReal y, PetscReal z)
{
  return (x-0.75)*x*(x+0.75) + (y-0.50)*y*(y+0.50) + (z-0.25)*z*(z+0.25);
}

#undef  __FUNCT__
#define __FUNCT__ "System"
PetscErrorCode System(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  PetscInt nen = p->nen;

  PetscReal x[3] = {0,0,0};
  IGAPointFormPoint(p,x);
  PetscScalar f = Function(x[0],x[1],x[2]);

  const PetscReal *N = (typeof(N)) p->shape[0];

  PetscInt a,b;
  for (a=0; a<nen; a++) {
    for (b=0; b<nen; b++) {
      K[a*nen+b] = N[a] * N[b];
    }
    F[a] = N[a] * f;
  }
  return 0;
}

#undef  __FUNCT__
#define __FUNCT__ "Error"
PetscErrorCode Error(IGAPoint p,const PetscScalar *U,PetscInt n,PetscScalar *S,void *ctx)
{
  PetscReal x[3] = {0,0,0};
  IGAPointFormPoint(p,x);
  PetscScalar f = Function(x[0],x[1],x[2]);

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

  PetscInt i;
  PetscInt dim = 2;
  PetscInt dof = 1;
  PetscInt N[3] = {16,16,16};
  PetscInt p[3] = { 2, 2, 2};
  PetscInt C[3] = {-1,-1,-1};
  PetscInt n1=3, n2=3, n3=3;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","L2Projection Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim","dimension",__FILE__,dim,&dim,NULL);CHKERRQ(ierr);
  n1 = n2 = n3 = dim;
  ierr = PetscOptionsIntArray("-N","number of elements",     __FILE__,N,&n1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-p","polynomial order",       __FILE__,p,&n2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-C","global continuity order",__FILE__,C,&n3,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (n1<3) N[2] = N[0]; if (n1<2) N[1] = N[0];
  if (n2<3) p[2] = p[0]; if (n2<2) p[1] = p[0];
  if (n3<3) C[2] = C[0]; if (n3<2) C[1] = C[0];
  for (i=0; i<dim; i++)  if (C[i] ==-1) C[i] = p[i] - 1;

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDim(iga,dim);CHKERRQ(ierr);
  ierr = IGASetDof(iga,dof);CHKERRQ(ierr);
  for (i=0; i<dim; i++) {
    IGAAxis axis;
    ierr = IGAGetAxis(iga,i,&axis);CHKERRQ(ierr);
    ierr = IGAAxisSetDegree(axis,p[i]);CHKERRQ(ierr);
    ierr = IGAAxisInitUniform(axis,N[i],-1.0,+1.0,C[i]);CHKERRQ(ierr);
  }
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  Mat A;
  Vec x,b;
  ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&b);CHKERRQ(ierr);
  ierr = IGASetFormSystem(iga,System,NULL);CHKERRQ(ierr);
  ierr = IGAComputeSystem(iga,A,b);CHKERRQ(ierr);

  KSP ksp;
  ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  PetscScalar error = 0;
  ierr = IGAComputeScalar(iga,x,1,&error,Error,NULL);CHKERRQ(ierr);
  error = PetscSqrtReal(PetscRealPart(error));
  PetscBool print_error = PETSC_FALSE;
  ierr = PetscOptionsGetBool(0,"-print_error",&print_error,0);CHKERRQ(ierr);
  if (print_error) {ierr = PetscPrintf(PETSC_COMM_WORLD,"L2 error = %G\n",error);CHKERRQ(ierr);}

  PetscBool draw = PETSC_FALSE;
  ierr = PetscOptionsGetBool(0,"-draw",&draw,0);CHKERRQ(ierr);
  if (draw && dim <= 2) {ierr = VecView(x,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
