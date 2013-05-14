#include "petiga.h"

PETSC_STATIC_INLINE
PetscReal DOT(PetscInt dim,const PetscReal a[],const PetscReal b[])
{
  PetscInt i; PetscReal s = 0.0;
  for (i=0; i<dim; i++) s += a[i]*b[i];
  return s;
}

#undef  __FUNCT__
#define __FUNCT__ "SystemGalerkin"
PetscErrorCode SystemGalerkin(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  PetscInt nen = p->nen;
  PetscInt dim = p->dim;
  const PetscReal *N0        = (typeof(N0)) p->shape[0];
  const PetscReal (*N1)[dim] = (typeof(N1)) p->shape[1];

  PetscInt a,b;
  for (a=0; a<nen; a++) {
    for (b=0; b<nen; b++)
      K[a*nen+b] = DOT(dim,N1[a],N1[b]);
    F[a] = N0[a] * 1.0;
  }
  return 0;
}

PETSC_STATIC_INLINE
PetscReal DEL2(PetscInt dim,const PetscReal a[dim][dim])
{
  PetscInt i; PetscReal s = 0.0;
  for (i=0; i<dim; i++) s += a[i][i];
  return s;
}

#undef  __FUNCT__
#define __FUNCT__ "SystemCollocation"
PetscErrorCode SystemCollocation(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  PetscInt nen = p->nen;
  PetscInt dim = p->dim;
  const PetscReal (*N2)[dim][dim] = (typeof(N2)) p->shape[2];

  PetscInt a;
  for (a=0; a<nen; a++)
    K[a] += -DEL2(dim,N2[a]);
  F[0] = 1.0;
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[]) {

  PetscErrorCode  ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  PetscBool draw = PETSC_FALSE;
  PetscBool save = PETSC_FALSE;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","Laplace Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-draw","If dim <= 2, then draw the solution to the screen",__FILE__,draw,&draw,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-save","Save the solution to file",                        __FILE__,save,&save,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDof(iga,1);CHKERRQ(ierr);
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  if (iga->dim < 1) {ierr = IGASetDim(iga,2);CHKERRQ(ierr);}
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  PetscInt dim,dir,side;
  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
  for (dir=0; dir<dim; dir++) {
    for (side=0; side<2; side++) {
      IGABoundary bnd;
      PetscScalar value = 1.0;
      ierr = IGAGetBoundary(iga,dir,side,&bnd);CHKERRQ(ierr);
      ierr = IGABoundarySetValue(bnd,0,value);CHKERRQ(ierr);
    }
  }

  Mat A;
  Vec x,b;
  ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&b);CHKERRQ(ierr);
  if (!iga->collocation) {
    ierr = IGASetUserSystem(iga,SystemGalerkin,NULL);CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_SPD,PETSC_TRUE);CHKERRQ(ierr);
  } else {
    ierr = IGASetUserSystem(iga,SystemCollocation,NULL);CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_FALSE);CHKERRQ(ierr);
  }
  ierr = IGAComputeSystem(iga,A,b);CHKERRQ(ierr);

  KSP ksp;
  ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  if (draw&&dim<3) {ierr = VecView(x,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}
  if (save)        {ierr = IGAWrite(iga,"PoissonGeometry.dat");CHKERRQ(ierr);}
  if (save)        {ierr = IGAWriteVec(iga,x,"PoissonSolution.dat");CHKERRQ(ierr);}

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
