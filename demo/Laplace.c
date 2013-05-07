/*
  This code solves the Laplace problem in one of the following ways:

  1) On the parametric unit domain [0,1]^dim (default)

    To solve on the parametric domain, do not specify a geometry
    file. You may change the discretization by altering the dimension
    of the space (-iga_dim), the number of uniform elements in each
    direction (-iga_elements), the polynomial order (-iga_degree),
    and the continuity (-iga_continuity).

  2) On a geometry

    If a geometry file is specified (-iga_geometry), the
    discretization will be what is read in from the geometry and is
    not editable from the command line.

  Note that the boundary conditions for this problem are such that
  the solution is always u(x)=1 (unit Dirichlet on the left side and
  free Neumann on the right). The error in the solution may be
  computed by using the -print_error command.

*/

#include "petiga.h"

PETSC_STATIC_INLINE
PetscReal DOT(PetscInt dim,const PetscReal a[],const PetscReal b[])
{
  PetscInt i; PetscReal s = 0.0;
  for (i=0; i<dim; i++) s += a[i]*b[i];
  return s;
}

PETSC_STATIC_INLINE
PetscReal DEL2(PetscInt dim,const PetscReal a[dim][dim])
{
  PetscInt i; PetscReal s = 0.0;
  for (i=0; i<dim; i++) s += a[i][i];
  return s;
}

#undef  __FUNCT__
#define __FUNCT__ "SystemGalerkin"
PetscErrorCode SystemGalerkin(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  PetscInt nen = p->nen;
  PetscInt dim = p->dim;

  const PetscReal (*N1)[dim];
  IGAPointGetShapeFuns(p,1,(const PetscReal **)&N1);

  PetscInt a,b;
  for (a=0; a<nen; a++) {
    for (b=0; b<nen; b++)
      K[a*nen+b] = DOT(dim,N1[a],N1[b]);
    F[a] = 0.0;
  }
  return 0;
}

#undef  __FUNCT__
#define __FUNCT__ "SystemCollocation"
PetscErrorCode SystemCollocation(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  PetscInt nen = p->nen;
  PetscInt dim = p->dim;

  const PetscReal (*N2)[dim][dim];
  IGAPointGetShapeFuns(p,2,(const PetscReal**)&N2);

  PetscInt a;
  for (a=0; a<nen; a++)
    K[a] += -DEL2(dim,N2[a]);
  F[0] = 0.0;

  return 0;
}

#undef  __FUNCT__
#define __FUNCT__ "Error"
PetscErrorCode Error(IGAPoint p,const PetscScalar *U,PetscInt n,PetscScalar *S,void *ctx)
{
  PetscScalar u;
  IGAPointFormValue(p,U,&u);
  PetscReal e = PetscAbsScalar(u - 1.0);
  S[0] = e*e;
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[]) {

  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  // Setup options

  PetscInt  i;
  PetscBool collocation = PETSC_FALSE;
  PetscBool print_error = PETSC_FALSE;
  PetscBool save = PETSC_FALSE;
  PetscBool draw = PETSC_FALSE;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","Laplace Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-collocation","Enable to use collocation",__FILE__,collocation,&collocation,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-print_error","Prints the L2 error of the solution",__FILE__,print_error,&print_error,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-save","Save the solution to file",__FILE__,save,&save,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-draw","If dim <= 2, then draw the solution to the screen",__FILE__,draw,&draw,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  // Initialize the discretization

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDof(iga,1);CHKERRQ(ierr);
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  if (iga->dim < 1) {ierr = IGASetDim(iga,3);CHKERRQ(ierr);}
  ierr = IGASetUp(iga);CHKERRQ(ierr);
  if (collocation) {ierr = IGASetUseCollocation(iga,PETSC_TRUE);CHKERRQ(ierr);}

  // Set boundary conditions

  PetscInt dim;
  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
  for (i=0; i<dim; i++) {
    IGABoundary bnd;
    ierr = IGAGetBoundary(iga,i,0,&bnd);CHKERRQ(ierr);
    ierr = IGABoundarySetValue(bnd,0,1.0);CHKERRQ(ierr);
    ierr = IGAGetBoundary(iga,i,1,&bnd);CHKERRQ(ierr);
    ierr = IGABoundarySetLoad(bnd,0,0.0);CHKERRQ(ierr);
  }

  // Assemble

  Mat A;
  Vec b,x;
  ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&b);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  if (!iga->collocation) {
    ierr = IGASetUserSystem(iga,SystemGalerkin,PETSC_NULL);CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
  } else {
    ierr = IGASetUserSystem(iga,SystemCollocation,PETSC_NULL);CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_FALSE);CHKERRQ(ierr);
  }
  ierr = IGAComputeSystem(iga,A,b);CHKERRQ(ierr);

  // Solve

  KSP ksp;
  ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  // Various post-processing options

  if (print_error) {
    PetscScalar error = 0;
    ierr = IGAFormScalar(iga,x,1,&error,Error,PETSC_NULL);CHKERRQ(ierr);
    error = PetscSqrtReal(PetscRealPart(error));
    ierr = PetscPrintf(PETSC_COMM_WORLD,"L2 error = %G\n",error);CHKERRQ(ierr);
  }
  if (save) {
    ierr = IGAWrite(iga,"LaplaceGeometry.dat");CHKERRQ(ierr);
    ierr = IGAWriteVec(iga,x,"LaplaceSolution.dat");CHKERRQ(ierr);
  }
  if (draw && dim <= 2) {
    ierr = VecView(x,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  }

  // Cleanup

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
