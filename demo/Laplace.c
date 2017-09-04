/*
  This code solves the Laplace problem in one of the following ways:

  1) On the parametric unit domain [0,1]^dim (default)

    To solve on the parametric domain, do not specify a geometry
    file. You may change the discretization by altering the dimension
    of the space (-iga_dim), the number of uniform elements in each
    direction (-iga_elements), the polynomial order (-iga_degree),
    and the continuity (-iga_continuity).

  2) On a geometry

    If a geometry file is specified (-iga_load), the
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

PetscErrorCode SystemGalerkin(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  PetscInt nen = p->nen;
  PetscInt dim = p->dim;
  const PetscReal (*N1)[dim] = (typeof(N1)) p->shape[1];

  PetscInt a,b;
  for (a=0; a<nen; a++) {
    for (b=0; b<nen; b++)
      K[a*nen+b] = DOT(dim,N1[a],N1[b]);
    F[a] = 0.0;
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

PetscErrorCode SystemCollocation(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  PetscInt nen = p->nen;
  PetscInt dim = p->dim;
  const PetscReal (*N2)[dim][dim] = (typeof(N2)) p->shape[2];

  PetscInt a;
  for (a=0; a<nen; a++)
    K[a] += -DEL2(dim,N2[a]);
  F[0] = 0.0;
  return 0;
}

PetscErrorCode Exact(IGAPoint p,PetscInt order,PetscScalar value[],void *ctx)
{
  value[0] = 1;
  return 0;
}

int main(int argc, char *argv[]) {

  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  // Setup options

  PetscBool print_error = PETSC_FALSE;
  PetscBool check_error = PETSC_FALSE;
  PetscBool save = PETSC_FALSE;
  PetscBool draw = PETSC_FALSE;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","Laplace Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-print_error","Prints the error of the solution",__FILE__,print_error,&print_error,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-check_error","Checks the error of the solution",__FILE__,check_error,&check_error,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-save","Save the solution to file",__FILE__,save,&save,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-draw","If dim <= 2, then draw the solution to the screen",__FILE__,draw,&draw,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  // Initialize the discretization

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDof(iga,1);CHKERRQ(ierr);
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  if (iga->dim<1) {ierr = IGASetDim(iga,2);CHKERRQ(ierr);}
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  // Set boundary conditions

  PetscInt dim,dir,side;
  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
  for (dir=0; dir<dim; dir++) {
    PetscScalar value = 1.0;
    PetscScalar load  = 0.0;
    ierr = IGASetBoundaryValue(iga,dir,side=0,0,value);CHKERRQ(ierr);
    ierr = IGASetBoundaryLoad (iga,dir,side=1,0,load );CHKERRQ(ierr);
  }

  // Assemble

  Mat A;
  Vec b,x;
  ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&b);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  if (!iga->collocation) {
    ierr = IGASetFormSystem(iga,SystemGalerkin,NULL);CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_SPD,PETSC_TRUE);CHKERRQ(ierr);
  } else {
    ierr = IGASetFormSystem(iga,SystemCollocation,NULL);CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_FALSE);CHKERRQ(ierr);
  }
  ierr = IGAComputeSystem(iga,A,b);CHKERRQ(ierr);

  // Solve

  KSP ksp;
  ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  // Various post-processing options

  PetscReal error;
  ierr = IGAComputeErrorNorm(iga,0,x,Exact,&error,NULL);CHKERRQ(ierr);
  
  if (print_error) {ierr = PetscPrintf(PETSC_COMM_WORLD,"Error = %g\n",(double)error);CHKERRQ(ierr);}
  if (check_error) {if (error>1e-3) SETERRQ1(PETSC_COMM_WORLD,1,"Error=%g\n",(double)error);}
  if (draw&&dim<3) {ierr = IGADrawVec(iga,x,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}
  if (save)        {ierr = IGAWrite   (iga,  "Laplace-geometry.dat");CHKERRQ(ierr);}
  if (save)        {ierr = IGAWriteVec(iga,x,"Laplace-solution.dat");CHKERRQ(ierr);}

  // Cleanup

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
