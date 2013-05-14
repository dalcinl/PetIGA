#include "petiga.h"

double CONST = 0.5;

#undef  __FUNCT__
#define __FUNCT__ "Galerkin1"
PetscErrorCode Galerkin1(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  PetscInt nen,dim;
  IGAPointGetSizes(p,0,&nen,0);
  IGAPointGetDims(p,&dim,0,0);

  const PetscReal *N0;
  IGAPointGetBasisFuns(p,0,&N0);
  const PetscReal *N1;
  IGAPointGetBasisFuns(p,1,&N1);

  PetscInt a,b,i;
  PetscScalar omega  = 2.0*PETSC_PI*CONST;
  PetscScalar omega2 = omega*omega;
  for (a=0; a<nen; a++) {
    for (b=0; b<nen; b++) {
      PetscScalar Kab = 0.0;
      for (i=0; i<dim; i++) Kab += N1[a*dim+i]*N1[b*dim+i];
      Kab += N0[a]*N0[b];
      K[a*nen+b] = Kab;
    }
    F[a] = (1.0+dim*omega2);
    for (i=0; i<dim; i++) F[a] *= sin(omega*p->point[i]);
    F[a] *= N0[a];
  }
  return 0;
}

#undef  __FUNCT__
#define __FUNCT__ "Collocation1"
PetscErrorCode Collocation1(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  PetscInt nen,dim,i;
  IGAPointGetSizes(p,0,&nen,0);
  IGAPointGetDims(p,&dim,0,0);

  PetscInt Nb[3] = {0,0,0};
  Nb[0] = p->parent->parent->axis[0]->nnp;
  Nb[1] = p->parent->parent->axis[1]->nnp;
  Nb[2] = p->parent->parent->axis[2]->nnp;

  const PetscReal *N0,(*N2)[dim][dim];
  IGAPointGetBasisFuns(p,0,(const PetscReal**)&N0);
  IGAPointGetBasisFuns(p,2,(const PetscReal**)&N2);

  PetscInt a;
  PetscBool Dirichlet=PETSC_FALSE;
  for (i=0; i<dim; i++) if (p->parent->ID[i] == 0 || p->parent->ID[i] == Nb[i]-1) Dirichlet=PETSC_TRUE;

  PetscScalar omega  = 2.0*PETSC_PI*CONST;
  PetscScalar omega2 = omega*omega;
  if(Dirichlet){
    for (a=0; a<nen; a++) K[a] = N0[a];
    F[0] = 0.0;
  }else{
    for (a=0; a<nen; a++){
      K[a] = N0[a];
      for (i=0; i<dim; i++) K[a] += -N2[a][i][i];
    }
    F[0] = (1.+dim*omega2);
    for (i=0; i<dim; i++) F[0] *= sin(omega*p->point[i]);
  }
  return 0;
}

#undef  __FUNCT__
#define __FUNCT__ "ErrorLaplace"
PetscErrorCode ErrorLaplace(IGAPoint p,const PetscScalar *U,PetscInt n,PetscScalar *S,void *ctx)
{
  PetscScalar ua,ue=1;
  IGAPointFormValue(p,U,&ua);

  PetscInt i,dim;
  IGAPointGetDims(p,&dim,0,0);
  for (i=0; i<dim; i++) ue *= sin(2*CONST*PETSC_PI*p->point[i]);

  PetscReal error = PetscAbsScalar(ua-ue);
  S[0] = error*error;
  S[1] = ue;
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[]) {

  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  // Initialize the discretization

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDof(iga,1);CHKERRQ(ierr);
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  // Set boundary conditions
  PetscInt  dim,i;
  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
  if(!iga->collocation){
    for (i=0; i<dim; i++) {
      IGABoundary bnd;
      ierr = IGAGetBoundary(iga,i,0,&bnd);CHKERRQ(ierr);
      ierr = IGABoundarySetValue(bnd,0,0.0);CHKERRQ(ierr);
      ierr = IGAGetBoundary(iga,i,1,&bnd);CHKERRQ(ierr);
      ierr = IGABoundarySetValue(bnd,0,0.0);CHKERRQ(ierr);
    }
  }

  // Assemble

  Mat A;
  Vec x,b;
  ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&b);CHKERRQ(ierr);
  if (iga->collocation){
    ierr = IGASetUserSystem(iga,Collocation1,NULL);CHKERRQ(ierr);
    ierr = IGAComputeSystem(iga,A,b);CHKERRQ(ierr);
  }else{
    ierr = IGASetUserSystem(iga,Galerkin1,NULL);CHKERRQ(ierr);
    ierr = IGAComputeSystem(iga,A,b);CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
  }

  // Solve

  KSP ksp;
  ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  // Various post-processing options

  iga->collocation = PETSC_FALSE;
  PetscScalar error[2] = {0,0};
  ierr = IGAFormScalar(iga,x,2,&error[0],ErrorLaplace,NULL);CHKERRQ(ierr);
  error[0] = PetscSqrtReal(PetscRealPart(error[0]));
  ierr = PetscPrintf(PETSC_COMM_WORLD,"L2 error = %.16e\n",error[0]);CHKERRQ(ierr);

  // Cleanup

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
