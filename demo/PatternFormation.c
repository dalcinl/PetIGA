#include "petiga.h"

/*

u_t = D_1 \nabla^2 u + f(u,v)
v_t = D_2 \nabla^2 v + g(u,v)

f(u,v) = alpha*u*(1-tau1*v**2) + v*(1-tau2*u);
g(u,v) = beta*v*(1+alpha*tau1/beta*u*v) + u*(gamma+tau2*v);

*/

typedef struct {
  PetscReal D1,D2;
  PetscReal alpha;
  PetscReal beta;
  PetscReal gamma;
  PetscReal tau1;
  PetscReal tau2;
} AppCtx;

#undef  __FUNCT__
#define __FUNCT__ "Function"
PetscErrorCode Function(IGAPoint point,PetscReal dt,PetscReal shift,
                        PetscReal t,const PetscScalar *U,const PetscScalar *V,
                        PetscScalar *Re,void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;

  PetscScalar uv_t[2],uv_0[2],uv_1[2][2];
  IGAPointInterpolate(point,0,V,&uv_t[0]);
  IGAPointInterpolate(point,0,U,&uv_0[0]);
  IGAPointInterpolate(point,1,U,&uv_1[0][0]);
  PetscScalar u_t = uv_t[0],    v_t = uv_t[1];    
  PetscScalar u   = uv_0[0],    v   = uv_0[1];    
  PetscScalar u_x = uv_1[0][0], v_x = uv_1[1][0];
  PetscScalar u_y = uv_1[0][1], v_y = uv_1[1][1];

  PetscReal D1    = user->D1;
  PetscReal D2    = user->D2;
  PetscReal alpha = user->alpha;
  PetscReal beta  = user->beta;
  PetscReal gamma = user->gamma;
  PetscReal tau1  = user->tau1;
  PetscReal tau2  = user->tau2;

  PetscReal f = alpha*u*(1-tau1*v*v) + v*(1-tau2*u);
  PetscReal g = beta*v*(1+alpha*tau1/beta*u*v) + u*(gamma+tau2*v);
  PetscReal (*N0)    = point->shape[0];
  PetscReal (*N1)[2] = (PetscReal (*)[2]) point->shape[1];
  PetscInt  a,nen=point->nen;
  PetscScalar (*R)[2] = (PetscScalar (*)[2])Re; 
  for (a=0; a<nen; a++) {
    PetscReal Na   = N0[a];
    PetscReal Na_x = N1[a][0];
    PetscReal Na_y = N1[a][1];
    PetscScalar Ru = Na*u_t + D1*(Na_x*u_x + Na_y*u_y) - Na*f;
    PetscScalar Rv = Na*v_t + D2*(Na_x*v_x + Na_y*v_y) - Na*g;
    R[a][0] = Ru;
    R[a][1] = Rv;
  }
  return 0;
}

#undef  __FUNCT__
#define __FUNCT__ "Jacobian"
PetscErrorCode Jacobian(IGAPoint point,PetscReal dt,PetscReal shift,
                        PetscReal t,const PetscScalar *U,const PetscScalar *V,
                        PetscScalar *Ke,void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;

  PetscScalar uv_0[2];
  IGAPointInterpolate(point,0,U,&uv_0[0]);
  PetscScalar u = uv_0[0], v = uv_0[1];    

  PetscReal D1    = user->D1;
  PetscReal D2    = user->D2;
  PetscReal alpha = user->alpha;
  PetscReal beta  = user->beta;
  PetscReal gamma = user->gamma;
  PetscReal tau1  = user->tau1;
  PetscReal tau2  = user->tau2;

  PetscReal f_u = alpha*(1-tau1*v*v) - tau2*v;
  PetscReal f_v = -2*alpha*tau1*u*v + (1-tau2*u);
  PetscReal g_u = alpha*tau1*v*v + (gamma+tau2*v);
  PetscReal g_v = (beta+2*alpha*tau1*u*v) + tau2*u;

  PetscReal (*N0)    = point->shape[0];
  PetscReal (*N1)[2] = (PetscReal (*)[2]) point->shape[1];
  PetscInt  i,j,a,b,nen=point->nen;
  PetscScalar (*K)[2][nen][2] = (PetscScalar (*)[2][nen][2])Ke;
  for (a=0; a<nen; a++) {
    PetscReal Na   = N0[a];
    PetscReal Na_x = N1[a][0];
    PetscReal Na_y = N1[a][1];
    for (b=0; b<nen; b++) {
      PetscReal Nb   = N0[b];
      PetscReal Nb_x = N1[b][0];
      PetscReal Nb_y = N1[b][1];
      PetscScalar Kab[2][2] = {{0,0},{0,0}};
      Kab[0][0] = shift*Na*Nb + D1*(Na_x*Nb_x + Na_y*Nb_y);
      Kab[1][1] = shift*Na*Nb + D2*(Na_x*Nb_x + Na_y*Nb_y);
      Kab[0][0] -= Na*f_u*Nb; Kab[0][1] -= Na*f_v*Nb;
      Kab[1][0] -= Na*g_u*Nb; Kab[1][1] -= Na*g_v*Nb;
      for (i=0;i<2;i++)
        for (j=0;j<2;j++)
          K[a][i][b][j] += Kab[i][j];
    }
  }
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[]) {

  PetscErrorCode  ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  AppCtx user;
  user.D1    = 1.0;
  user.D2    = 1.0;
  user.alpha = 1.0;
  user.beta  = 1.0;
  user.gamma = 1.0;
  user.tau1  = 1.0;
  user.tau2  = 1.0; 

  PetscInt i;
  PetscInt dim = 2;
  PetscInt dof = 2;
  PetscInt N[2] = {16,16}, nN = 2; 
  PetscInt p[2] = { 2, 2}, np = 2;
  PetscInt C[2] = {-1,-1}, nC = 2;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","PatternFormation Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-N","number of elements",     __FILE__,N,&nN,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-p","polynomial order",       __FILE__,p,&np,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-C","global continuity order",__FILE__,C,&nC,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (nN == 1) N[1] = N[0];
  if (np == 1) p[1] = p[0];
  if (nC == 1) C[1] = C[0];
  if (C[0] == -1) C[0] = p[0]-1;
  if (C[1] == -1) C[1] = p[1]-1;

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDim(iga,dim);CHKERRQ(ierr);
  ierr = IGASetDof(iga,dof);CHKERRQ(ierr);
  for (i=0; i<dim; i++) {
    IGAAxis axis;
    ierr = IGAGetAxis(iga,i,&axis);CHKERRQ(ierr);
    ierr = IGAAxisSetPeriodic(axis,PETSC_TRUE);CHKERRQ(ierr);
    ierr = IGAAxisSetOrder(axis,p[i]);CHKERRQ(ierr);
    ierr = IGAAxisInitUniform(axis,p[i],C[i],N[i],0.0,1.0);CHKERRQ(ierr);
  }
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  ierr = IGASetUserIFunction(iga,Function,&user);CHKERRQ(ierr);
  ierr = IGASetUserIJacobian(iga,Jacobian,&user);CHKERRQ(ierr);

  TS ts;
  ierr = IGACreateTS(iga,&ts);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,10000,1.0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,1e-3);CHKERRQ(ierr);

  ierr = TSSetType(ts,TSTHETA);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  PetscReal t; Vec U;
  ierr = IGACreateVec(iga,&U);CHKERRQ(ierr);
  ierr = VecSetRandom(U,PETSC_NULL);CHKERRQ(ierr);
  ierr = TSSolve(ts,U,&t);CHKERRQ(ierr);

  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
