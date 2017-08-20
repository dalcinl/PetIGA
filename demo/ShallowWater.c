#include "petiga.h"

typedef struct {
  PetscReal z,a,s;
  PetscReal gravity;
  PetscReal viscosity;
  TS ts; /* XXX */
} AppCtx;

PetscReal Tau(PetscReal dt, PetscScalar u0, PetscScalar u1, PetscInt nen, const PetscReal (*N1)[2])
{
  PetscInt a,i;
  PetscScalar u[2] = {0};
  u[0] = u0; u[1] = u1;
  PetscReal tau1 = 0, tau2 = 0;
  for (a=0; a<nen; a++) {
    PetscReal sum = 0;
    for (i=0; i<2; i++) {
      sum += u[i]*N1[a][i];
    }
    tau1 += PetscAbs(sum);
  }
  tau1 = 1/tau1;
  tau2 = dt/2;
  PetscReal tau = 1/sqrt(1/(tau1*tau1)+1/(tau2*tau2));
  return tau;
}

PetscScalar SUPG(PetscReal tau, PetscReal W[3], PetscScalar R[3])
{
  PetscInt i;
  PetscScalar r = 0;
  for (i=0; i<3; i++)
    r += W[i]*R[i];
  r *= tau;
  return r;
}

PetscErrorCode Residual(IGAPoint p,
                        PetscReal shift,const PetscScalar *V,
                        PetscReal t,const PetscScalar *U,
                        PetscScalar *Re,void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;
  PetscReal g  = user->gravity;
  PetscReal nu = user->viscosity; (void)nu;
  PetscReal dt; TSGetTimeStep(user->ts,&dt);

  PetscScalar s_t[3],s[3],grad_s[3][2];
  IGAPointFormValue(p,V,&s_t[0]);
  IGAPointFormValue(p,U,&s[0]);
  IGAPointFormGrad (p,U,&grad_s[0][0]);

  PetscScalar h_t  = s_t[0], h  = s[0];
  PetscScalar hu_t = s_t[1], hu = s[1], u = hu/h;
  PetscScalar hv_t = s_t[2], hv = s[2], v = hv/h;

  PetscScalar h_x  = grad_s[0][0], h_y  = grad_s[0][1];
  PetscScalar hu_x = grad_s[1][0], hu_y = grad_s[1][1];
  PetscScalar hv_x = grad_s[2][0], hv_y = grad_s[2][1];

  const PetscReal *N0,(*N1)[2];
  IGAPointGetShapeFuns(p,0,(const PetscReal**)&N0);
  IGAPointGetShapeFuns(p,1,(const PetscReal**)&N1);

  PetscScalar res[3];
  res[0] = h_t  + ( hu_x                       ) + ( hv_y                       );
  res[1] = hu_t + ( (-u*u+g*h)*h_x + 2*u*hu_x  ) + ( -u*v*h_y + v*hu_y + u*hv_y );
  res[2] = hv_t + ( -u*v*h_x + v*hu_x + u*hv_x ) + ( (-v*v+g*h)*h_y + 2*v*hv_y  );

  PetscReal tau = Tau(dt,u,v,p->nen,N1);

  PetscScalar (*R)[3] = (PetscScalar (*)[3])Re;
  PetscInt a,nen = p->nen;
  for (a=0; a<nen; a++) {
    PetscReal Na   = N0[a];
    PetscReal Na_x = N1[a][0];
    PetscReal Na_y = N1[a][1];
    PetscScalar Rh,Rhu,Rhv;
    /* ----- */

    Rh  = Na * res[0];
    Rhu = Na * res[1];
    Rhv = Na * res[2];

    PetscReal W0[3];
    W0[0] = 0;
    W0[1] = (-u*u+g*h) * Na_x + (-u*v)     * Na_y;
    W0[2] = (-u*v)     * Na_x + (-v*v+g*h) * Na_y;

    PetscReal W1[3];
    W1[0] = (1)   * Na_x;
    W1[1] = (2*u) * Na_x + (v) * Na_y;
    W1[2] = (v)   * Na_x;

    PetscReal W2[3];
    W2[0] =                (1) * Na_y;
    W2[1] =                (u) * Na_y;
    W2[2] = (u) * Na_x + (2*v) * Na_y;

    Rh  += SUPG(tau,W0,res);
    Rhu += SUPG(tau,W1,res);
    Rhv += SUPG(tau,W2,res);

    /* ----- */

    R[a][0] = Rh;
    R[a][1] = Rhu;
    R[a][2] = Rhv;
  }

  return 0;
}

/*
PetscErrorCode Tangent(IGAPoint p,
                       PetscReal shift,const PetscScalar *V,
                       PetscReal t,const PetscScalar *U,
                       PetscScalar *Ke,void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;
  return 0;
}
*/

PetscErrorCode FormInitialCondition(AppCtx *user,IGA iga,PetscReal t,Vec U)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecZeroEntries(U);CHKERRQ(ierr);

  DM da;
  ierr = IGACreateNodeDM(iga,3,&da);CHKERRQ(ierr);
  DMDALocalInfo info;
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  PetscScalar ***s;
  ierr = DMDAVecGetArrayDOF(da,U,&s);CHKERRQ(ierr);

  PetscReal imax = (info.mx-1), Lx = 1.0;
  PetscReal jmax = (info.my-1), Ly = 1.0;
  PetscInt  i,j;
  for(j=info.ys;j<info.ys+info.ym;j++){
    for(i=info.xs;i<info.xs+info.xm;i++){
      PetscReal x = (i/imax)*Lx;
      PetscReal y = (j/jmax)*Ly;
      PetscReal r2 = (x-0.5)*(x-0.5) + (y-0.5)*(y-0.5);
      PetscReal h = user->z + user->a*exp(-r2/(2*user->s*user->s));
      s[j][i][0] = h;
      s[j][i][1] = 0;
      s[j][i][2] = 0;
    }
  }

  ierr = DMDAVecRestoreArrayDOF(da,U,&s);CHKERRQ(ierr);

  ierr = DMDestroy(&da);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

int main(int argc, char *argv[]) {

  PetscErrorCode  ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  AppCtx user;

  user.z         = 1.0;
  user.a         = 0.5;
  user.s         = 0.1;

  user.gravity   = 1.0;
  user.viscosity = 0.0;

  PetscBool W[2] = {PETSC_FALSE,PETSC_FALSE}; PetscInt nW = 2;
  PetscInt  N[2] = {64,64}, nN = 2;
  PetscInt  p[2] = { 2, 2}, np = 2;
  PetscInt  C[2] = {-1,-1}, nC = 2;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","ShallowWater Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsBoolArray("-W","periodicity",            __FILE__,W,&nW,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray ("-N","number of elements",     __FILE__,N,&nN,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray ("-p","polynomial order",       __FILE__,p,&np,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray ("-C","global continuity order",__FILE__,C,&nC,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (nW == 1) W[1] = W[0];
  if (nN == 1) N[1] = N[0];
  if (np == 1) p[1] = p[0];
  if (nC == 1) C[1] = C[0];
  if (C[0] == -1) C[0] = p[0]-1;
  if (C[1] == -1) C[1] = p[1]-1;

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDim(iga,2);CHKERRQ(ierr);
  ierr = IGASetDof(iga,3);CHKERRQ(ierr);

  ierr = IGASetFieldName(iga,0,"h");CHKERRQ(ierr);
  ierr = IGASetFieldName(iga,1,"hu");CHKERRQ(ierr);
  ierr = IGASetFieldName(iga,2,"hv");CHKERRQ(ierr);

  PetscInt i,j;
  IGAAxis axis;
  for (i=0; i<2; i++) {
    ierr = IGAGetAxis(iga,i,&axis);CHKERRQ(ierr);
    ierr = IGAAxisSetDegree(axis,p[i]);CHKERRQ(ierr);
    ierr = IGAAxisSetPeriodic(axis,W[i]);CHKERRQ(ierr);
    ierr = IGAAxisInitUniform(axis,N[i],0.0,1.0,C[i]);CHKERRQ(ierr);
    if (W[i]) continue;
    for (j=0; j<2; j++) {
      IGASetBoundaryValue(iga,i,j,1,0.0);
      IGASetBoundaryValue(iga,i,j,2,0.0);
    }
  }

  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  ierr = IGASetFormIFunction(iga,Residual,&user);CHKERRQ(ierr);
  ierr = IGASetFormIJacobian(iga,IGAFormIJacobianFD,&user);CHKERRQ(ierr);

  TS ts;
  ierr = IGACreateTS(iga,&ts);CHKERRQ(ierr);
  user.ts = ts;

  ierr = TSSetType(ts,TSALPHA);CHKERRQ(ierr);
  //ierr = TSAlphaSetRadius(ts,0.5);CHKERRQ(ierr);
  PetscReal dx = 1.0/PetscMax(N[0],N[1]);
  PetscReal dt = 0.5 * dx/sqrt(user.gravity*user.a);
  ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,1000.0);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  PetscReal t=0; Vec U;
  ierr = IGACreateVec(iga,&U);CHKERRQ(ierr);
  ierr = FormInitialCondition(&user,iga,t,U);CHKERRQ(ierr);
  ierr = TSSolve(ts,U);CHKERRQ(ierr);

  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
