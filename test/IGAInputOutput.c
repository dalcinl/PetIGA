#include "petiga.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[]) {

  MPI_Comm       comm;
  IGA            iga,iga1;
  PetscViewer    viewer;
  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  comm = PETSC_COMM_WORLD;
  ierr = IGACreate(comm,&iga);CHKERRQ(ierr);
  {
    PetscInt  i;
    PetscInt  dim = 3;
    PetscInt  dof = 1;
    PetscBool b[3] = {PETSC_FALSE, PETSC_FALSE, PETSC_FALSE};
    PetscInt  N[3] = {16,16,16};
    PetscInt  p[3] = { 2, 2, 2};
    PetscInt  C[3] = {-1,-1,-1};
    PetscInt  n0=3, n1=3, n2=3, n3=3;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","InputOutput Options","IGA");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-iga_dim","dimension",__FILE__,dim,&dim,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-iga_dof","dofs/node",__FILE__,dof,&dof,NULL);CHKERRQ(ierr);
    n0 = n1 = n2 = n3 = dim;
    ierr = PetscOptionsBoolArray("-periodic","periodicity",     __FILE__,b,&n0,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsIntArray ("-N","number of elements",     __FILE__,N,&n1,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsIntArray ("-p","polynomial order",       __FILE__,p,&n2,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsIntArray ("-C","global continuity order",__FILE__,C,&n3,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    if (n0<3) b[2] = b[0]; if (n0<2) b[1] = b[0];
    if (n1<3) N[2] = N[0]; if (n1<2) N[1] = N[0];
    if (n2<3) p[2] = p[0]; if (n2<2) p[1] = p[0];
    if (n3<3) C[2] = C[0]; if (n3<2) C[1] = C[0];
    for (i=0; i<dim; i++)  if (C[i] ==-1) C[i] = p[i] - 1;
    ierr = IGASetDim(iga,dim);CHKERRQ(ierr);
    ierr = IGASetDof(iga,dof);CHKERRQ(ierr);
    for (i=0; i<dim; i++) {
      IGAAxis axis;
      ierr = IGAGetAxis(iga,i,&axis);CHKERRQ(ierr);
      ierr = IGAAxisSetPeriodic(axis,b[i]);CHKERRQ(ierr);
      ierr = IGAAxisSetDegree(axis,p[i]);CHKERRQ(ierr);
      ierr = IGAAxisInitUniform(axis,N[i],0.0,1.0,C[i]);CHKERRQ(ierr);
    }
  }
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  ierr = IGAWrite(iga,"iga.dat");CHKERRQ(ierr);
  ierr = IGAWrite(iga,"iga.dat");CHKERRQ(ierr); /* just for testing */

  ierr = IGACreate(comm,&iga1);CHKERRQ(ierr);
  ierr = IGARead(iga1,"iga.dat");CHKERRQ(ierr);
  ierr = IGASetUp(iga1);CHKERRQ(ierr);
  ierr = IGARead(iga1,"iga.dat");CHKERRQ(ierr);  /* just for testing */
  ierr = IGASetUp(iga1);CHKERRQ(ierr);
  ierr = IGADestroy(&iga1);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(comm,"iga.dat",FILE_MODE_WRITE,&viewer);
  ierr = IGASave(iga,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = IGACreate(comm,&iga1);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(comm,"iga.dat",FILE_MODE_READ,&viewer);
  ierr = IGALoad(iga1,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = IGASetUp(iga1);CHKERRQ(ierr);
  ierr = IGADestroy(&iga1);CHKERRQ(ierr);

  {
    Vec         vec;
    PetscInt    size,bs;
    PetscScalar value;
    ierr = IGACreateVec(iga,&vec);CHKERRQ(ierr);
    ierr = VecSet(vec,1.0);CHKERRQ(ierr);
    ierr = IGAWriteVec(iga,vec,"igavec.dat");CHKERRQ(ierr);
    ierr = VecSet(vec,0.0);CHKERRQ(ierr);
    ierr = IGAReadVec (iga,vec,"igavec.dat");CHKERRQ(ierr);
    ierr = VecGetSize(vec,&size);CHKERRQ(ierr);
    ierr = VecSum(vec,&value);CHKERRQ(ierr);
    if ((PetscReal)size != PetscRealPart(value))
      SETERRQ(comm,PETSC_ERR_PLIB,"Bad data in file");
    ierr = VecDestroy(&vec);CHKERRQ(ierr);

    ierr = IGAGetDof(iga,&bs);CHKERRQ(ierr);
    ierr = VecCreate(comm,&vec);CHKERRQ(ierr);
    ierr = VecSetBlockSize(vec,bs);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(comm,"igavec.dat",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = VecLoad(vec,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = VecGetSize(vec,&size);CHKERRQ(ierr);
    ierr = VecSum(vec,&value);CHKERRQ(ierr);
    if ((PetscReal)size != PetscRealPart(value))
      SETERRQ(comm,PETSC_ERR_PLIB,"Bad Vec data in file");
    ierr = VecDestroy(&vec);CHKERRQ(ierr);
  }

  {
    Mat         mat;
    Vec         diag,diag2;
    PetscInt    r,rstart,rend;
    PetscScalar *d;
    PetscBool   match = PETSC_FALSE;

    ierr = IGACreateVec(iga,&diag);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(diag,&rstart,&rend);CHKERRQ(ierr);
    ierr = VecGetArray(diag,&d);CHKERRQ(ierr);
    for (r=rstart; r<rend; r++) d[r-rstart] = (PetscReal)r;
    ierr = VecRestoreArray(diag,&d);CHKERRQ(ierr);

    ierr = IGACreateMat(iga,&mat);CHKERRQ(ierr);
    ierr = MatDiagonalSet(mat,diag,INSERT_VALUES);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(comm,"igamat.dat",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = MatView(mat,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = MatDestroy(&mat);CHKERRQ(ierr);

    ierr = IGACreateMat(iga,&mat);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(comm,"igamat.dat",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = MatLoad(mat,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = IGACreateVec(iga,&diag2);CHKERRQ(ierr);
    ierr = MatGetDiagonal(mat,diag2);CHKERRQ(ierr);
    ierr = MatDestroy(&mat);CHKERRQ(ierr);

    ierr = VecEqual(diag,diag2,&match);;CHKERRQ(ierr);
    if (!match) SETERRQ(comm,PETSC_ERR_PLIB,"Bad Mat data in file");
    ierr = VecDestroy(&diag);CHKERRQ(ierr);
    ierr = VecDestroy(&diag2);CHKERRQ(ierr);
  }

  ierr = IGADestroy(&iga);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
 }
