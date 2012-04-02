#include "petiga.h"

#undef  __FUNCT__
#define __FUNCT__ "System"
PetscErrorCode System(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  PetscInt  i,dim = p->dim;
  PetscInt  a,b,nen=p->nen;
  PetscReal *N1 = p->shape[1];
  for (a=0; a<nen; a++) {
    for (b=0; b<nen; b++) {
      PetscScalar Kab = 0.0;
      for (i=0; i<dim; i++)
        Kab += N1[a*dim+i]*N1[b*dim+i];
      K[a*nen+b] = Kab;
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

  PetscInt  i,j;
  PetscInt  dim = 3;
  PetscInt  dof = 1;
  PetscBool t[3] = {PETSC_FALSE, PETSC_FALSE, PETSC_FALSE};
  PetscInt  N[3] = {16,16,16}; 
  PetscInt  p[3] = { 2, 2, 2};
  PetscInt  C[3] = {-1,-1,-1};
  PetscInt  n0=3, n1=3, n2=3, n3=3;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","Laplace Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim","dimension",__FILE__,dim,&dim,PETSC_NULL);CHKERRQ(ierr);
  n0 = n1 = n2 = n3 = dim;
  ierr = PetscOptionsBoolArray("-periodic","periodicity",     __FILE__,t,&n0,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray ("-N","number of elements",     __FILE__,N,&n1,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray ("-p","polynomial order",       __FILE__,p,&n2,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray ("-C","global continuity order",__FILE__,C,&n3,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (n0<3) t[2] = t[0]; if (n0<2) t[1] = t[0];
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
    ierr = IGAAxisSetPeriodic(axis,t[i]);CHKERRQ(ierr);
    ierr = IGAAxisSetDegree(axis,p[i]);CHKERRQ(ierr);
    ierr = IGAAxisInitUniform(axis,N[i],0.0,1.0,C[i]);CHKERRQ(ierr);
    for (j=0; j<2; j++) {
      IGABoundary bnd;
      PetscInt    field = 0;
      PetscScalar value = j ? 1.0 : 0.0;
      ierr = IGAGetBoundary(iga,i,j,&bnd);CHKERRQ(ierr);
      ierr = IGABoundarySetValue(bnd,field,value);CHKERRQ(ierr);
    }
  }
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  Mat A;
  Vec x,b;
  ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&b);CHKERRQ(ierr);
  ierr = IGAFormSystem(iga,A,b,System,PETSC_NULL);CHKERRQ(ierr);
  
  KSP ksp;
  ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  PetscBool draw = PETSC_FALSE;
  ierr = PetscOptionsGetBool(0,"-draw",&draw,0);CHKERRQ(ierr);
  if (draw && dim <= 2) {
    ierr = VecView(x,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  }

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
