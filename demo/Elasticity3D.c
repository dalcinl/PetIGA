#include "petiga.h"

typedef struct {
  PetscReal lambda,mu;
} AppCtx;

#undef  __FUNCT__
#define __FUNCT__ "System"
PetscErrorCode System(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;

  PetscReal lambda = user->lambda;
  PetscReal mu = user->mu;

  PetscReal (*N1)[3] = (PetscReal (*)[3]) p->shape[1];
  PetscInt a,b,nen=p->nen;
  PetscScalar (*Kl)[3][nen][3] = (PetscScalar (*)[3][nen][3])K;

  for (a=0; a<nen; a++) {
    PetscReal Na_x = N1[a][0];
    PetscReal Na_y = N1[a][1];
    PetscReal Na_z = N1[a][2];
    for (b=0; b<nen; b++) {
      PetscReal Nb_x = N1[b][0];
      PetscReal Nb_y = N1[b][1];
      PetscReal Nb_z = N1[b][2];
      Kl[a][0][b][0] = Na_x*Nb_x*(lambda + 2*mu) + mu*(Na_y*Nb_y + Na_z*Nb_z);
      Kl[a][0][b][1] = Na_x*Nb_y*lambda + Na_y*Nb_x*mu;
      Kl[a][0][b][2] = Na_x*Nb_z*lambda + Na_z*Nb_x*mu;
      Kl[a][1][b][0] = Na_x*Nb_y*mu + Na_y*Nb_x*lambda;
      Kl[a][1][b][1] = Na_y*Nb_y*(lambda + 2*mu) + mu*(Na_z*Nb_z + Na_x*Nb_x*mu);
      Kl[a][1][b][2] = Na_y*Nb_z*lambda + Na_z*Nb_y*mu;
      Kl[a][2][b][0] = Na_x*Nb_z*mu + Na_z*Nb_x*lambda;
      Kl[a][2][b][1] = Na_y*Nb_z*mu + Na_z*Nb_y*lambda;
      Kl[a][2][b][2] = mu*(Na_x*Nb_x + Na_y*Nb_y) + Na_z*Nb_z*(lambda + 2*mu);
    }
    F[a] = 0.0;
  }
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[]) {

  PetscErrorCode  ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  // Define problem parameters
  AppCtx user;
  user.lambda = 1.0;
  user.mu = 1.0;

  // Setup discretization
  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDim(iga,3);CHKERRQ(ierr);
  ierr = IGASetDof(iga,3);CHKERRQ(ierr);
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  // Set boundary conditions
  IGABoundary bnd;
  ierr = IGAGetBoundary(iga,0,0,&bnd);CHKERRQ(ierr);
  ierr = IGABoundarySetValue(bnd,0,0.0);CHKERRQ(ierr);
  ierr = IGABoundarySetValue(bnd,1,0.0);CHKERRQ(ierr);
  ierr = IGABoundarySetValue(bnd,2,0.0);CHKERRQ(ierr);
  ierr = IGAGetBoundary(iga,0,1,&bnd);CHKERRQ(ierr);
  ierr = IGABoundarySetValue(bnd,0,1.0);CHKERRQ(ierr);

  // Create linear system
  Mat A;
  Vec x,b;
  ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&b);CHKERRQ(ierr);
  ierr = IGASetUserSystem(iga,System,&user);CHKERRQ(ierr);
  ierr = IGAFormSystem(iga,A,b);CHKERRQ(ierr);
  
  // Solve
  KSP ksp;
  ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  
  // Dump solution vector
  MPI_Comm        comm;
  PetscViewer     viewer;
  ierr = PetscObjectGetComm((PetscObject)x,&comm);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(comm,"solution.dat",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(x,viewer);CHKERRQ(ierr);
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  // Cleanup
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
