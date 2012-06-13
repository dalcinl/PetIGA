#include "petiga.h"

typedef struct {
  PetscReal E,nu,lambda,mu;
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

      Kl[a][0][b][0] += (lambda+2.0*mu)*Na_x*Nb_x + mu*(Na_y*Nb_y + Na_z*Nb_z);
      Kl[a][1][b][1] += (lambda+2.0*mu)*Na_y*Nb_y + mu*(Na_x*Nb_y + Na_z*Nb_z);
      Kl[a][2][b][2] += (lambda+2.0*mu)*Na_z*Nb_z + mu*(Na_y*Nb_y + Na_x*Nb_x);
			  
      Kl[a][0][b][1] += lambda*Na_x*Nb_y + mu*Na_y*Nb_x;
      Kl[a][1][b][0] += lambda*Na_y*Nb_x + mu*Na_x*Nb_y;
      
      Kl[a][0][b][2] += lambda*Na_x*Nb_z + mu*Na_z*Nb_x;
      Kl[a][2][b][0] += lambda*Na_z*Nb_x + mu*Na_x*Nb_z;
      
      Kl[a][1][b][2] += lambda*Na_y*Nb_z + mu*Na_z*Nb_y;
      Kl[a][2][b][1] += lambda*Na_z*Nb_y + mu*Na_y*Nb_z;      
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

  PetscInt N[3] = {16,16,16}, nN = 3; 
  PetscInt p[3] = { 2, 2, 2}, np = 3;
  PetscInt C[3] = {-1,-1,-1}, nC = 3;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","Poisson2D Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-N","number of elements",     __FILE__,N,&nN,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-p","polynomial order",       __FILE__,p,&np,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-C","global continuity order",__FILE__,C,&nC,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (nN == 1) N[2] = N[1] = N[0]; if (nN == 2) N[2] = N[0];
  if (np == 1) p[2] = p[1] = p[0]; if (np == 2) p[2] = p[0];
  if (nC == 1) C[2] = C[1] = C[0]; if (nC == 2) C[2] = C[0];
  if (C[0] == -1) C[0] = p[0]-1;
  if (C[1] == -1) C[1] = p[1]-1;
  if (C[2] == -1) C[2] = p[2]-1;

  AppCtx user;
  user.E = 75.0e6;
  user.nu = 0.25;
  user.lambda = user.E*user.nu/(1.0+user.nu)/(1.0-2.0*user.nu);
  user.mu = 0.5*user.E/(1.0+user.nu);

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDim(iga,3);CHKERRQ(ierr);
  ierr = IGASetDof(iga,3);CHKERRQ(ierr);
  
  PetscInt i;
  for (i=0; i<3; i++) {
    IGAAxis axis;
    ierr = IGAGetAxis(iga,i,&axis);CHKERRQ(ierr);
    ierr = IGAAxisSetDegree(axis,p[i]);CHKERRQ(ierr);
    ierr = IGAAxisInitUniform(axis,N[i],0.0,1.0,C[i]);CHKERRQ(ierr);
  }

  IGABoundary bnd;
  ierr = IGAGetBoundary(iga,0,0,&bnd);CHKERRQ(ierr);
  ierr = IGABoundarySetValue(bnd,0,0.0);CHKERRQ(ierr);
  ierr = IGABoundarySetValue(bnd,1,0.0);CHKERRQ(ierr);
  ierr = IGABoundarySetValue(bnd,2,0.0);CHKERRQ(ierr);
  ierr = IGAGetBoundary(iga,0,1,&bnd);CHKERRQ(ierr);
  ierr = IGABoundarySetValue(bnd,0,1.0);CHKERRQ(ierr);
  ierr = IGABoundarySetValue(bnd,1,0.0);CHKERRQ(ierr);
  ierr = IGABoundarySetValue(bnd,2,0.0);CHKERRQ(ierr);

  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  Mat A;
  Vec x,b;
  ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&b);CHKERRQ(ierr);
  ierr = IGAFormSystem(iga,A,b,System,&user);CHKERRQ(ierr);
  
  KSP ksp;
  ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
