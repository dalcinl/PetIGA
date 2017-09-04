/*
  This code solves the following system of PDEs

     u_t = D_1 \nabla^2 u + f(u,v)
     v_t = D_2 \nabla^2 v + g(u,v)

     f(u,v) = alpha*u*(1-tau1*v**2) + v*(1-tau2*u);
     g(u,v) = beta*v*(1+alpha*tau1/beta*u*v) + u*(gamma+tau2*v);

  and highlights how to use implicit/explicit capabilities of PetIGA.
*/
#include "petiga.h"

#define EXPLICIT 1
typedef struct {
  PetscBool IMPLICIT;
  PetscReal delta;
  PetscReal D1,D2;
  PetscReal alpha;
  PetscReal beta;
  PetscReal gamma;
  PetscReal tau1;
  PetscReal tau2;
} AppCtx;

PetscErrorCode Function(IGAPoint p,
                        PetscReal shift,const PetscScalar *V,
                        PetscReal t,const PetscScalar *U1,
                        PetscReal t0,const PetscScalar *U0,
                        PetscScalar *Re,void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;
  PetscBool IMPLICIT = user->IMPLICIT;

  PetscReal delta = user->delta;
  PetscReal D1    = user->D1;
  PetscReal D2    = user->D2;
  PetscReal alpha = user->alpha;
  PetscReal beta  = user->beta;
  PetscReal gamma = user->gamma;
  PetscReal tau1  = user->tau1;
  PetscReal tau2  = user->tau2;

  PetscInt nen = p->nen;
  PetscScalar (*R)[2] = (PetscScalar (*)[2])Re;

  PetscScalar uv_t[2],uv_0[2],uv_1[2][2];
  IGAPointFormValue(p,V,&uv_t[0]);
  if (IMPLICIT)
    IGAPointFormValue(p,U1,&uv_0[0]);
  else
    IGAPointFormValue(p,U0,&uv_0[0]);
  IGAPointFormGrad(p,U1,&uv_1[0][0]);
  PetscReal u_t = uv_t[0],    v_t = uv_t[1];
  PetscReal u   = uv_0[0],    v   = uv_0[1];
  PetscReal u_x = uv_1[0][0], v_x = uv_1[1][0];
  PetscReal u_y = uv_1[0][1], v_y = uv_1[1][1];

  PetscReal f = alpha*u*(1-tau1*v*v) + v*(1-tau2*u);
  PetscReal g = beta*v*(1+alpha*tau1/beta*u*v) + u*(gamma+tau2*v);

  const PetscReal *N0      = p->shape[0];
  const PetscReal (*N1)[2] = (const PetscReal(*)[2]) p->shape[1];

  PetscInt a;
  for (a=0; a<nen; a++) {
    PetscReal Na   = N0[a];
    PetscReal Na_x = N1[a][0];
    PetscReal Na_y = N1[a][1];
    PetscReal Ru = Na*u_t + delta*D1*(Na_x*u_x + Na_y*u_y) - Na*f;
    PetscReal Rv = Na*v_t + delta*D2*(Na_x*v_x + Na_y*v_y) - Na*g;
    R[a][0] = Ru;
    R[a][1] = Rv;
  }
  return 0;
}

PetscErrorCode Jacobian(IGAPoint p,
                        PetscReal shift,const PetscScalar *V,
                        PetscReal t,const PetscScalar *U1,
                        PetscReal t0,const PetscScalar *U0,
                        PetscScalar *Ke,void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;
  PetscBool IMPLICIT = user->IMPLICIT;

  PetscReal delta = user->delta;
  PetscReal D1    = user->D1;
  PetscReal D2    = user->D2;
  PetscReal alpha = user->alpha;
  PetscReal beta  = user->beta;
  PetscReal gamma = user->gamma;
  PetscReal tau1  = user->tau1;
  PetscReal tau2  = user->tau2;

  PetscInt nen = p->nen;
  PetscScalar (*K)[2][nen][2] = (typeof(K)) Ke;

  PetscReal f_u=0,f_v=0;
  PetscReal g_u=0,g_v=0;
  if (IMPLICIT) {
    PetscScalar uv_0[2];
    IGAPointFormValue(p,U1,&uv_0[0]);
    PetscReal u = uv_0[0];
    PetscReal v = uv_0[1];
    f_u = alpha*(1-tau1*v*v) - tau2*v;
    f_v = -2*alpha*tau1*u*v + (1-tau2*u);
    g_u = alpha*tau1*v*v + (gamma+tau2*v);
    g_v = (beta+2*alpha*tau1*u*v) + tau2*u;
  }

  const PetscReal *N0,(*N1)[2];
  IGAPointGetShapeFuns(p,0,(const PetscReal**)&N0);
  IGAPointGetShapeFuns(p,1,(const PetscReal**)&N1);

  PetscInt  a,b,i,j;
  for (a=0; a<nen; a++) {
    PetscReal Na   = N0[a];
    PetscReal Na_x = N1[a][0];
    PetscReal Na_y = N1[a][1];
    for (b=0; b<nen; b++) {
      PetscReal Nb   = N0[b];
      PetscReal Nb_x = N1[b][0];
      PetscReal Nb_y = N1[b][1];
      PetscReal Kab[2][2] = {{0,0},{0,0}};
      Kab[0][0] = shift*Na*Nb + delta*D1*(Na_x*Nb_x + Na_y*Nb_y);
      Kab[1][1] = shift*Na*Nb + delta*D2*(Na_x*Nb_x + Na_y*Nb_y);
      if (IMPLICIT) {
        Kab[0][0] -= Na*f_u*Nb; Kab[0][1] -= Na*f_v*Nb;
        Kab[1][0] -= Na*g_u*Nb; Kab[1][1] -= Na*g_v*Nb;
        for (i=0;i<2;i++)
          for (j=0;j<2;j++)
            K[a][i][b][j] += Kab[i][j];
      } else {
        K[a][0][b][0] += Kab[0][0];
        K[a][1][b][1] += Kab[1][1];
      }
    }
  }
  return 0;
}

int main(int argc, char *argv[]) {

  PetscErrorCode  ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  AppCtx user;
  user.delta = 0.0045;
  user.D1    =  0.500;
  user.D2    =  1.000;
  user.alpha =  0.899;
  user.beta  = -0.910;
  user.gamma = -user.alpha;
  user.tau1  =  0.020;
  user.tau2  =  0.200;

  user.IMPLICIT = PETSC_FALSE;

  PetscInt i;
  PetscInt dim = 2;
  PetscInt dof = 2;
  PetscInt N[2] = {32,32}, nN = 2;
  PetscInt p[2] = { 2, 2}, np = 2;
  PetscInt C[2] = {-1,-1}, nC = 2;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","PatternFormation Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-implicit","Treat all terms implicitly",__FILE__,user.IMPLICIT,&user.IMPLICIT,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-N","number of elements",     __FILE__,N,&nN,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-p","polynomial order",       __FILE__,p,&np,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-C","global continuity order",__FILE__,C,&nC,NULL);CHKERRQ(ierr);
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
    ierr = IGAAxisSetDegree(axis,p[i]);CHKERRQ(ierr);
    ierr = IGAAxisInitUniform(axis,N[i],-1.0,+1.0,C[i]);CHKERRQ(ierr);
  }
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  ierr = IGASetFormIEFunction(iga,Function,&user);CHKERRQ(ierr);
  ierr = IGASetFormIEJacobian(iga,Jacobian,&user);CHKERRQ(ierr);

  PetscReal h  = PetscMin(2.0/N[0],2.0/N[1]);
  PetscReal dt = h/user.delta/15.0;

  TS ts;
  ierr = IGACreateTS(iga,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSBEULER);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts,120);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetTime(ts,0.0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
  if (!user.IMPLICIT) {
    ierr = TSSetProblemType(ts,TS_LINEAR);CHKERRQ(ierr);
  }
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  Vec U;
  ierr = IGACreateVec(iga,&U);CHKERRQ(ierr);
  ierr = VecSetRandom(U,NULL);CHKERRQ(ierr);
  ierr = VecScale(U,1.);CHKERRQ(ierr);
  ierr = TSSolve(ts,U);CHKERRQ(ierr);

  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
