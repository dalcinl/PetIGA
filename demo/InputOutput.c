#include "petiga.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[]) {

  PetscErrorCode  ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  PetscInt  i;
  PetscInt  dim = 3;
  PetscInt  dof = 1;
  PetscBool b[3] = {PETSC_FALSE, PETSC_FALSE, PETSC_FALSE};
  PetscInt  N[3] = {16,16,16}; 
  PetscInt  p[3] = { 2, 2, 2};
  PetscInt  C[3] = {-1,-1,-1};
  PetscInt  n0=3, n1=3, n2=3, n3=3;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","InputOutput Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim","dimension",__FILE__,dim,&dim,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dof","dofs/node",__FILE__,dof,&dof,PETSC_NULL);CHKERRQ(ierr);
  n0 = n1 = n2 = n3 = dim;
  ierr = PetscOptionsBoolArray("-periodic","periodicity",     __FILE__,b,&n0,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray ("-N","number of elements",     __FILE__,N,&n1,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray ("-p","polynomial order",       __FILE__,p,&n2,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray ("-C","global continuity order",__FILE__,C,&n3,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (n0<3) b[2] = b[0]; if (n0<2) b[1] = b[0];
  if (n1<3) N[2] = N[0]; if (n1<2) N[1] = N[0];
  if (n2<3) p[2] = p[0]; if (n2<2) p[1] = p[0];
  if (n3<3) C[2] = C[0]; if (n3<2) C[1] = C[0];
  for (i=0; i<dim; i++)  if (C[i] ==-1) C[i] = p[i] - 1;
  
  IGA iga; IGAAxis axis;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDim(iga,dim);CHKERRQ(ierr);
  ierr = IGASetDof(iga,dof);CHKERRQ(ierr);
  for (i=0; i<dim; i++) {
    ierr = IGAGetAxis(iga,i,&axis);CHKERRQ(ierr);
    ierr = IGAAxisSetPeriodic(axis,b[i]);CHKERRQ(ierr);
    ierr = IGAAxisInitUniform(axis,p[i],C[i],N[i],0.0,1.0);CHKERRQ(ierr);
  }
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);
  
  MPI_Comm comm;
  ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
  ierr = IGAWrite(iga,"iga.dat");CHKERRQ(ierr);
  IGA iganew;
  ierr = IGACreate(comm,&iganew);CHKERRQ(ierr);
  ierr = IGARead(iganew,"iga.dat");CHKERRQ(ierr);
  ierr = IGASetUp(iganew);CHKERRQ(ierr);

  ierr = IGADestroy(&iganew);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
 }
