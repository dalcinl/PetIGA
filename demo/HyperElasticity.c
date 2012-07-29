#include "petiga.h"

typedef struct {
  PetscReal lambda,mu;
} AppCtx;

#undef  __FUNCT__
#define __FUNCT__ "Residual"
PetscErrorCode Residual(IGAPoint pnt,const PetscScalar *U,PetscScalar *Re,void *ctx)
{    
  // This is the Residual for HyperElasticity without Neumann conditions and body forces. 
  //   R(w,u;u)=a(w,u;u)-L(w)~0
  AppCtx *user = (AppCtx *)ctx;

  PetscReal lambda = user->lambda;
  PetscReal mu = user->mu;

  // Interpolate the solution and gradient given U
  PetscScalar u[3];
  PetscScalar grad_u[3][3];
  IGAPointInterpolate(pnt,0,U,&u[0]);
  IGAPointInterpolate(pnt,1,U,&grad_u[0][0]);

  // Get basis functions and gradients
  PetscReal (*N1)[3] = (PetscReal (*)[3]) pnt->shape[1];

  // Compute the deformation gradient,its determinant, and inverse
  PetscScalar F[3][3],Finv[3][3];
  PetscScalar J;

  F[0][0] = 1.0 + grad_u[0][0]; F[0][1] = grad_u[0][1]; F[0][2] = grad_u[0][2]; 
  F[1][0] = grad_u[1][0]; F[1][1] = 1.0 + grad_u[1][1]; F[1][2] = grad_u[1][2]; 
  F[2][0] = grad_u[2][0]; F[2][1] = grad_u[2][1]; F[2][2] = 1.0 + grad_u[2][2]; 

  J  = F[0][0]*(F[1][1]*F[2][2] - F[1][2]*F[2][1]);
  J -= F[0][1]*(F[1][0]*F[2][2] - F[1][2]*F[2][0]);
  J += F[0][2]*(F[1][0]*F[2][1] - F[1][1]*F[2][0]);

  Finv[0][0] =  (F[1][1]*F[2][2]-F[2][1]*F[1][2])/J;
  Finv[0][1] = -(F[1][0]*F[2][2]-F[1][2]*F[2][0])/J;
  Finv[0][2] =  (F[1][0]*F[2][1]-F[2][0]*F[1][1])/J;
  Finv[1][0] = -(F[0][1]*F[2][2]-F[0][2]*F[2][1])/J;
  Finv[1][1] =  (F[0][0]*F[2][2]-F[0][2]*F[2][0])/J;
  Finv[1][2] = -(F[0][0]*F[2][1]-F[2][0]*F[0][1])/J;
  Finv[2][0] =  (F[0][1]*F[1][2]-F[0][2]*F[1][1])/J;
  Finv[2][1] = -(F[0][0]*F[1][2]-F[1][0]*F[0][2])/J;
  Finv[2][2] =  (F[0][0]*F[1][1]-F[1][0]*F[0][1])/J;

  // C^-1 = (F^T F)^-1 = F^-1 F^-T
  PetscScalar Cinv[3][3];
  Cinv[0][0] = Finv[0][0]*Finv[0][0] + Finv[0][1]*Finv[0][1] + Finv[0][2]*Finv[0][2];
  Cinv[0][1] = Finv[0][0]*Finv[1][0] + Finv[0][1]*Finv[1][1] + Finv[0][2]*Finv[1][2];
  Cinv[0][2] = Finv[0][0]*Finv[2][0] + Finv[0][1]*Finv[2][1] + Finv[0][2]*Finv[2][2];
  Cinv[1][0] = Finv[1][0]*Finv[0][0] + Finv[1][1]*Finv[0][1] + Finv[1][2]*Finv[0][2];
  Cinv[1][1] = Finv[1][0]*Finv[1][0] + Finv[1][1]*Finv[1][1] + Finv[1][2]*Finv[1][2];
  Cinv[1][2] = Finv[1][0]*Finv[2][0] + Finv[1][1]*Finv[2][1] + Finv[1][2]*Finv[2][2];
  Cinv[2][0] = Finv[2][0]*Finv[0][0] + Finv[2][1]*Finv[0][1] + Finv[2][2]*Finv[0][2];
  Cinv[2][1] = Finv[2][0]*Finv[1][0] + Finv[2][1]*Finv[1][1] + Finv[2][2]*Finv[1][2];
  Cinv[2][2] = Finv[2][0]*Finv[2][0] + Finv[2][1]*Finv[2][1] + Finv[2][2]*Finv[2][2];

  // Stress tensor
  PetscScalar S[3][3];
  lambda = (0.5*lambda)*(J*J-1.); // redefine lambda to save FLOPS
  S[0][0] = lambda*Cinv[0][0] + mu*(1.0-Cinv[0][0]);
  S[0][1] = lambda*Cinv[0][1] + mu*(Cinv[0][1]);
  S[0][2] = lambda*Cinv[0][2] + mu*(Cinv[0][2]);
  S[1][0] = lambda*Cinv[1][0] + mu*(Cinv[1][0]);
  S[1][1] = lambda*Cinv[1][1] + mu*(1.0-Cinv[1][1]);
  S[1][2] = lambda*Cinv[1][2] + mu*(Cinv[1][2]);
  S[2][0] = lambda*Cinv[2][0] + mu*(Cinv[2][0]);
  S[2][1] = lambda*Cinv[2][1] + mu*(Cinv[2][1]);
  S[2][2] = lambda*Cinv[2][2] + mu*(1.0-Cinv[2][2]);

  // Piola stress tensor
  PetscScalar P[3][3];
  P[0][0] = F[0][0]*S[0][0] + F[0][1]*S[1][0] + F[0][2]*S[2][0];
  P[0][1] = F[0][0]*S[0][1] + F[0][1]*S[1][1] + F[0][2]*S[2][1];
  P[0][2] = F[0][0]*S[0][2] + F[0][1]*S[1][2] + F[0][2]*S[2][2];
  P[1][0] = F[1][0]*S[0][0] + F[1][1]*S[1][0] + F[1][2]*S[2][0];
  P[1][1] = F[1][0]*S[0][1] + F[1][1]*S[1][1] + F[1][2]*S[2][1];
  P[1][2] = F[1][0]*S[0][2] + F[1][1]*S[1][2] + F[1][2]*S[2][2];
  P[2][0] = F[2][0]*S[0][0] + F[2][1]*S[1][0] + F[2][2]*S[2][0];
  P[2][1] = F[2][0]*S[0][1] + F[2][1]*S[1][1] + F[2][2]*S[2][1];
  P[2][2] = F[2][0]*S[0][2] + F[2][1]*S[1][2] + F[2][2]*S[2][2];

  // Put together the residual
  PetscScalar (*R)[3] = (PetscScalar (*)[3])Re;
  PetscInt a,nen=pnt->nen;
  for (a=0; a<nen; a++) {
    PetscReal Na_x  = N1[a][0];
    PetscReal Na_y  = N1[a][1];
    PetscReal Na_z  = N1[a][2];
    R[a][0] = Na_x*P[0][0]+Na_y*P[0][1]+Na_z*P[0][2]; 
    R[a][1] = Na_x*P[1][0]+Na_y*P[1][1]+Na_z*P[1][2];
    R[a][2] = Na_x*P[2][0]+Na_y*P[2][1]+Na_z*P[2][2];
  }

  return 0;
}


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[]) 
{
  // Initialization of PETSc
  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);
  
  // Application specific data
  AppCtx user;
  user.lambda = 1.0;
  user.mu     = 1.0;

  // Initialize the discretization
  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDof(iga,3);CHKERRQ(ierr);
  ierr = IGASetDim(iga,3);CHKERRQ(ierr);
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  // Set boundary conditions
  IGABoundary bnd;
  ierr = IGAGetBoundary(iga,0,0,&bnd);CHKERRQ(ierr); // u = [0,0,0] @ x = [0,:,:]
  ierr = IGABoundarySetValue(bnd,0,0.0);CHKERRQ(ierr);
  ierr = IGABoundarySetValue(bnd,1,0.0);CHKERRQ(ierr);
  ierr = IGABoundarySetValue(bnd,2,0.0);CHKERRQ(ierr);
  ierr = IGAGetBoundary(iga,0,1,&bnd);CHKERRQ(ierr); // ux = 1 @ x = [1,:,:]
  ierr = IGABoundarySetValue(bnd,0,1.0);CHKERRQ(ierr);

  // Setup the nonlinear solver
  SNES snes;
  ierr = IGASetUserFunction(iga,Residual,&user);CHKERRQ(ierr);
  ierr = IGACreateSNES(iga,&snes);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  // Solve
  Vec U;
  ierr = IGACreateVec(iga,&U);CHKERRQ(ierr);
  ierr = SNESSolve(snes,PETSC_NULL,U);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
