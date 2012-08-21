#include "petiga.h"

#undef  __FUNCT__
#define __FUNCT__ "SystemLaplace"
PetscErrorCode SystemLaplace(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  PetscInt nen,dim;
  IGAPointGetSizes(p,&nen,0,&dim);

  const PetscReal *N1;
  IGAPointGetShapeFuns(p,1,&N1);

  PetscInt a,b,i;
  for (a=0; a<nen; a++) {
    for (b=0; b<nen; b++) {
      PetscScalar Kab = 0.0;
      for (i=0; i<dim; i++)
        Kab += N1[a*dim+i]*N1[b*dim+i];
      K[a*nen+b] = Kab;
    }
  }
  return 0;
}

#undef  __FUNCT__
#define __FUNCT__ "SystemPoisson"
PetscErrorCode SystemPoisson(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  PetscInt nen,dim;
  IGAPointGetSizes(p,&nen,0,&dim);

  const PetscReal *N;
  IGAPointGetShapeFuns(p,0,&N);
  const PetscReal *N1;
  IGAPointGetShapeFuns(p,1,&N1);

  PetscInt a,b,i;
  for (a=0; a<nen; a++) {
    for (b=0; b<nen; b++) {
      PetscScalar Kab = 0.0;
      for (i=0; i<dim; i++)
        Kab += N1[a*dim+i]*N1[b*dim+i];
      K[a*nen+b] = Kab;
    }
    F[a] = 1.0*N[a];
  }
  return 0;
}

#undef  __FUNCT__
#define __FUNCT__ "SystemCollocation"
PetscErrorCode SystemCollocation(IGAColPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  PetscInt nen,dim;
  IGAColPointGetSizes(p,&nen,0,&dim);

  const PetscReal *N0,(*N1)[dim],(*N2)[dim][dim];
  IGAColPointGetBasisFuns(p,0,(const PetscReal**)&N0);
  IGAColPointGetBasisFuns(p,1,(const PetscReal**)&N1);
  IGAColPointGetBasisFuns(p,2,(const PetscReal**)&N2);
  
  PetscInt a,i;
  for (a=0; a<nen; a++) 
    for (i=0; i<dim; i++) {
      K[a] += -N2[a][i][i];
    }
      
  return 0;
}

#undef  __FUNCT__
#define __FUNCT__ "ErrorLaplace"
PetscErrorCode ErrorLaplace(IGAPoint p,const PetscScalar *U,PetscInt n,PetscScalar *S,void *ctx)
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

  /*
    This code solves the Laplace problem in one of the following ways:
    
    1) On the parametric unit domain [0,1]^dim (default)
    
      To solve on the parametric domain, do not specify a geometry
      file. You may change the discretization by altering the
      dimension of the space (-dim), the number of uniform elements in
      each direction (-iga_elements), the polynomial order
      (-iga_degree), and the continuity (-iga_continuity). Note that
      the boundary conditions for this problem are such that the
      solution is always u(x)=1 (unit Dirichlet on the left side and
      free Neumann on the right). The error in the solution may be
      computed by using the -print_error command.

    2) On a geometry

      If a geometry file is specified (-iga_geometry), then the code
      will solve the Poisson problem on this geometry. The forcing is
      set to 1 and we have 0 Dirichlet conditions everywhere. We use
      this mode to test geometries as the solution should display the
      same symmetry as the geometry. The discretization will be what
      is read in from the geometry and is not editable from the
      commandline.

   */

  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  // Setup options

  PetscInt  i; 
  PetscInt  dim = 3; 
  PetscBool print_error = PETSC_FALSE; 
  PetscBool draw = PETSC_FALSE; 
  PetscBool Collocation = PETSC_FALSE;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","Laplace Options","IGA");CHKERRQ(ierr); 
  ierr = PetscOptionsInt("-dim","dimension",__FILE__,dim,&dim,PETSC_NULL);CHKERRQ(ierr); 
  ierr = PetscOptionsBool("-print_error","Prints the L2 error of the solution",__FILE__,print_error,&print_error,PETSC_NULL);CHKERRQ(ierr); 
  ierr = PetscOptionsBool("-draw","If dim <= 2, then draw the solution to the screen",__FILE__,draw,&draw,PETSC_NULL);CHKERRQ(ierr); 
  ierr = PetscOptionsBool("-collocation","Enable to use collocation",__FILE__,Collocation,&Collocation,PETSC_NULL);CHKERRQ(ierr); 
  ierr = PetscOptionsEnd();CHKERRQ(ierr); 

  // Initialize the discretization

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDof(iga,1);CHKERRQ(ierr);
  ierr = IGASetDim(iga,dim);CHKERRQ(ierr);
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  // Set boundary conditions

  if (iga->geometry) {
    for(i=0; i<dim; i++) {
      IGABoundary bnd;
      ierr = IGAGetBoundary(iga,i,0,&bnd);CHKERRQ(ierr);
      ierr = IGABoundarySetValue(bnd,0,0.0);CHKERRQ(ierr);
      ierr = IGAGetBoundary(iga,i,1,&bnd);CHKERRQ(ierr);
      ierr = IGABoundarySetValue(bnd,0,0.0);CHKERRQ(ierr);
    }
  }else{
    for (i=0; i<dim; i++) {
      IGABoundary bnd;
      ierr = IGAGetBoundary(iga,i,0,&bnd);CHKERRQ(ierr);
      ierr = IGABoundarySetValue(bnd,0,1.0);CHKERRQ(ierr);
    }
  }
  
  // Assemble

  Mat A;
  Vec x,b;
  ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&b);CHKERRQ(ierr);
  if (Collocation){
    ierr = IGAColSetUserSystem(iga,SystemCollocation,PETSC_NULL);CHKERRQ(ierr); 
    ierr = IGAColComputeSystem(iga,A,b);CHKERRQ(ierr);
  }else{
    if (iga->geometry){ 
      ierr = IGASetUserSystem(iga,SystemPoisson,PETSC_NULL);CHKERRQ(ierr); 
    }else{
      ierr = IGASetUserSystem(iga,SystemLaplace,PETSC_NULL);CHKERRQ(ierr); 
    }
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

  if (iga->geometry) {
    MPI_Comm        comm;
    PetscViewer     viewer;
    ierr = PetscObjectGetComm((PetscObject)x,&comm);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(comm,"solution.dat",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = VecView(x,viewer);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  if (print_error && !iga->geometry) {
    PetscScalar error = 0;
    ierr = IGAFormScalar(iga,x,1,&error,ErrorLaplace,PETSC_NULL);CHKERRQ(ierr);
    error = PetscSqrtReal(PetscRealPart(error));
    ierr = PetscPrintf(PETSC_COMM_WORLD,"L2 error = %G\n",error);CHKERRQ(ierr);
  }

  if (draw && dim <= 2) {ierr = VecView(x,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}

  // Cleanup

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
