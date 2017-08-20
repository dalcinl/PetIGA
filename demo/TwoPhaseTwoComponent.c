/*
  This code solves multiphase, multicomponent flow in porous media
  (two phase, two component). Thanks to Bilal Saad for his
  contribution.

  keywords: transient, vector, implicit, nonlinear, dimension
  independent, boundary integrals
 */

#include "petiga.h"

typedef struct {
  IGA       iga;
  PetscReal rholw,porosity,Pr,n,Slr,Sgr,H,Mh,T,mul,mug,Mw,Dlh,k;
} AppCtx;

typedef struct {
  PetscReal Pl,rholh;
} Field;

PetscReal SEC_PER_YEAR = 365.0*24.0*3600.0;

void EquationOfState(PetscInt dim,PetscScalar Pl,PetscScalar Pl_t,PetscScalar rholh,PetscScalar rholh_t,PetscScalar *rholh_x,
                     PetscScalar *Sl,PetscScalar *Sl_t,PetscScalar *krl,PetscScalar *krg,
                     PetscScalar *rhogh,PetscScalar *rhogh_t,PetscScalar *Pg_x,AppCtx *user)
{
  // VanGenuchten, Henry
  PetscScalar R     = 8.314e-5; // [J/(K-mol)]
  PetscScalar Pr    = user->Pr;
  PetscScalar n     = user->n, m = 1.-1./n;
  PetscScalar Slr   = user->Slr;
  PetscScalar Sgr   = user->Sgr;
  PetscScalar H     = user->H;
  PetscScalar Mh    = user->Mh;
  PetscScalar T     = user->T;
  PetscScalar Pc    = rholh/(H*Mh)-Pl;
  PetscScalar Pc_t  = rholh_t/(H*Mh)-Pl_t;
  PetscScalar Sle   = 1;
  PetscScalar Sle_t = 0;
  if(PetscRealPart(rholh/H/Mh-Pl) > 0.0) {
    Sle   = pow(pow((Pc/Pr),n)+1.,-m);
    Sle_t = -m*n/Pr*pow(pow((Pc/Pr),n)+1.,-m-1.)*pow(Pc/Pr,n-1.)*Pc_t;
  }
  *Sl      = (1.-Slr-Sgr)*Sle+Slr;
  *Sl_t    = (1.-Slr-Sgr)*Sle_t;
  *krl     = sqrt(Sle)*pow(1.-pow(1.-pow(Sle,1./m),m),2);
  *krg     = sqrt(1.-Sle)*pow(1.-pow(Sle,1./m),2*m);
  *rhogh_t = rholh_t/(R*T*H);
  *rhogh   = rholh  /(R*T*H);
  PetscInt i;
  for(i=0;i<dim;i++) Pg_x[i] = rholh_x[i]/Mh*H;
  return;
}

PetscErrorCode LeftInjectionResidual(IGAPoint p,
                                     PetscReal shift,const PetscScalar *V,
                                     PetscReal t,const PetscScalar *U,
                                     PetscScalar *R,void *ctx)
{
  PetscInt a,nen;
  IGAPointGetSizes(p,&nen,0,0);

  const PetscReal *N0;
  IGAPointGetShapeFuns(p,0,(const PetscReal**)&N0);

  PetscScalar Qh = 0;
  if(t <= 5e5) Qh = -5.57e-6; // inflow

  PetscScalar (*Re)[2] = (PetscScalar (*)[2])R;
  for (a=0; a<nen; a++) {
    Re[a][0] = 0.0;
    Re[a][1] = N0[a]*Qh;
  }
  return 0;
}

PetscErrorCode Residual(IGAPoint p,
                        PetscReal shift,const PetscScalar *V,
                        PetscReal t,const PetscScalar *U,
                        PetscScalar *R,void *ctx)
{
  if (p->atboundary)
    return LeftInjectionResidual(p,shift,V,t,U,R,ctx);

  AppCtx *user = (AppCtx *)ctx;
  PetscScalar rholw = user->rholw;
  PetscScalar porosity = user->porosity;
  PetscScalar mul = user->mul;
  PetscScalar mug = user->mug;
  PetscScalar Mh  = user->Mh;
  PetscScalar Mw  = user->Mw;
  PetscScalar Dlh = user->Dlh;
  PetscScalar k   = user->k;

  PetscInt a,i,nen,dim;
  IGAPointGetSizes(p,&nen,0,0);
  IGAPointGetDims(p,&dim,NULL,NULL);

  PetscScalar sol_t[2],sol[2],sol_x[2][dim];
  IGAPointFormValue(p,V,&sol_t[0]);
  IGAPointFormValue(p,U,&sol[0]);
  IGAPointFormGrad (p,U,&sol_x[0][0]);

  const PetscReal *N0,(*N1)[dim];
  IGAPointGetShapeFuns(p,0,(const PetscReal**)&N0);
  IGAPointGetShapeFuns(p,1,(const PetscReal**)&N1);

  PetscScalar  Pl      = sol[0];    // liquid pressure
  PetscScalar  Pl_t    = sol_t[0];
  PetscScalar *Pl_x    = sol_x[0];
  PetscScalar  rholh   = sol[1];    // dissolved hydrogen
  PetscScalar  rholh_t = sol_t[1];
  PetscScalar *rholh_x = sol_x[1];

  PetscScalar Sl,Sl_t,krl,krg,rhogh,rhogh_t,Pg_x[dim];
  EquationOfState(dim,Pl,Pl_t,rholh,rholh_t,rholh_x,&Sl,&Sl_t,&krl,&krg,&rhogh,&rhogh_t,Pg_x,user);
  PetscScalar Sg = 1.-Sl, Sg_t = -Sl_t;
  PetscScalar cl = Sl*(rholh/Mh+rholw/Mw);

  PetscScalar den = rholh/Mh+rholw/Mw;
  PetscScalar jlw[dim],ql[dim],qg[dim],g=0;
  for(i=0;i<dim;i++) {
    if(i==dim-1) g = 0;
    PetscScalar X_x = (rholh_x[i]/Mh)/den - (rholh/Mh)/(den*den)*(rholh_x[i]/Mh);
    jlw[i] = porosity*Mh*cl*Dlh*X_x;
    ql[i]  = -k*krl/mul*(Pl_x[i]-rholw*g);
    qg[i]  = -k*krg/mug*(Pg_x[i]);
  }

  PetscScalar (*Re)[2] = (PetscScalar (*)[2])R;
  for (a=0; a<nen; a++) {
    PetscScalar gN_dot_ql = 0,gN_dot_qg = 0,gN_dot_jlw = 0;
    for(i=0;i<dim;i++){
      gN_dot_ql += N1[a][i]*ql[i];
      gN_dot_qg += N1[a][i]*qg[i];
      gN_dot_jlw += N1[a][i]*jlw[i];
    }
    Re[a][0]  = porosity*N0[a]*Sl_t*rholw-(rholw*gN_dot_ql + gN_dot_jlw);
    Re[a][1]  = porosity*N0[a]*(Sl_t*rholh+Sl*rholh_t+Sg_t*rhogh+Sg*rhogh_t);
    Re[a][1] -= (rholh*gN_dot_ql + rhogh*gN_dot_qg - gN_dot_jlw);
  }
  return 0;
}

PetscErrorCode LeftInjectionJacobian(IGAPoint p,
                                     PetscReal shift,const PetscScalar *V,
                                     PetscReal t,const PetscScalar *U,
                                     PetscScalar *J,void *ctx)
{
  // for now use the option -snes_fd_color for the Jacobian
  return 0;
}

PetscErrorCode Jacobian(IGAPoint p,
                        PetscReal shift,const PetscScalar *V,
                        PetscReal t,const PetscScalar *U,
                        PetscScalar *J,void *ctx)
{
  if (p->atboundary)
    return LeftInjectionJacobian(p,shift,V,t,U,J,ctx);
  // for now use the option -snes_fd_color for the Jacobian
  return 0;
}

PetscErrorCode InitialCondition(IGA iga,Vec U,AppCtx *user)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  DM da;
  ierr = IGACreateNodeDM(iga,2,&da);CHKERRQ(ierr);
  Field **u;
  ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);
  DMDALocalInfo info;
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);

  PetscInt i,j;
  for(i=info.xs;i<info.xs+info.xm;i++){
    for(j=info.ys;j<info.ys+info.ym;j++){
      u[j][i].Pl    = 10.;
      u[j][i].rholh = 0;
    }
  }
  ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr);
  ierr = DMDestroy(&da);;CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal t,Vec U,void *mctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  AppCtx *user = (AppCtx *)mctx;
  IGAProbe prb;
  IGA      iga = user->iga;
  PetscInt dim = iga->dim;

  if(step == 0){
    PetscPrintf(PETSC_COMM_WORLD,"#%11s %12s %12s %12s %12s %12s %12s\n","Time","dt","Pl(xL)","Pg(xL)","Sg(xL)","fluxw(xR)","fluxh(xR)");
  }

  // Pl,Pg,Sg computed at left middle
  ierr = IGAProbeCreate(iga,U,&prb);CHKERRQ(ierr);
  PetscReal point[2] = {0,10};

  PetscScalar sol[2];
  ierr = IGAProbeSetPoint(prb,point);CHKERRQ(ierr);
  ierr = IGAProbeFormValue(prb,&sol[0]);CHKERRQ(ierr);
  PetscScalar Pl,Pg,Sg;
  Pl = sol[0];
  Pg = sol[1]/(user->Mh*user->H);
  PetscScalar Pc  = Pg-Pl;
  PetscScalar Sle = 1;
  if(PetscRealPart(Pc) > 0.0) {
    Sle   = pow(pow((Pc/user->Pr),user->n)+1.,-1.+1./user->n);
  }
  Sg = 1.-((1.-user->Slr-user->Sgr)*Sle+user->Slr);

  PetscReal dt;
  TSGetTimeStep(ts,&dt);
  PetscPrintf(PETSC_COMM_WORLD,"%.6e %.6e %.6e %.6e %.6e ",t,dt,Pl,Pg,Sg);

  // fluxw,fluxh computed at right middle
  point[0] = 200;
  PetscScalar sol_x[2][2];
  ierr = IGAProbeSetPoint(prb,point);CHKERRQ(ierr);
  ierr = IGAProbeFormValue(prb,&sol[0]);CHKERRQ(ierr);
  ierr = IGAProbeFormGrad (prb,&sol_x[0][0]);CHKERRQ(ierr);
  Pl = sol[0];
  PetscScalar *Pl_x = sol_x[0];
  Pg = sol[1]/(user->Mh*user->H);
  Pc  = Pg-Pl;
  Sle = 1;
  if(PetscRealPart(Pc) > 0.0) {
    Sle   = pow(pow((Pc/user->Pr),user->n)+1.,-1.+1./user->n);
  }
  Sg = 1.-((1.-user->Slr-user->Sgr)*Sle+user->Slr);
  PetscScalar  rholh   = sol[1];
  PetscScalar *rholh_x = sol_x[1];

  PetscScalar Sl,Sl_t,krl,krg,rhogh,rhogh_t,Pg_x[dim];
  EquationOfState(dim,Pl,0,rholh,0,rholh_x,&Sl,&Sl_t,&krl,&krg,&rhogh,&rhogh_t,Pg_x,user);
  PetscScalar fluxw = -user->k*krl/user->mul*(Pl_x[0]);
  PetscScalar fluxh = -user->k*krg/user->mug*(Pg_x[0]);

  PetscPrintf(PETSC_COMM_WORLD,"%.6e %.6e\n",fluxw,fluxh);

  ierr = IGAProbeDestroy(&prb);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

int main(int argc, char *argv[]) {

  PetscErrorCode  ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  AppCtx user;
  user.rholw    = 1000;
  user.porosity = 0.15;
  user.Pr       = 20;
  user.n        = 1.49;
  user.Slr      = 0.4;
  user.Sgr      = 0.;
  user.T        = 303;
  user.H        = 0.765;
  user.Mw       = 1e-2;
  user.Mh       = 2e-3;
  user.mul      = 1e-8/SEC_PER_YEAR;
  user.mug      = 9e-11/SEC_PER_YEAR;
  user.Dlh      = 3e-9*SEC_PER_YEAR;
  user.k        = 5e-20;
  PetscInt dim = 2, p = 1, N = 200, L = 200;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"r_","2c2p Options","2c2p");CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  IGA         iga;
  IGAAxis     axis;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDim(iga,dim);CHKERRQ(ierr);
  ierr = IGASetDof(iga,2);CHKERRQ(ierr);

  ierr = IGAGetAxis(iga,0,&axis);CHKERRQ(ierr);
  ierr = IGAAxisSetDegree(axis,p);CHKERRQ(ierr);
  ierr = IGAAxisInitUniform(axis,N,0.0,L,0);CHKERRQ(ierr);
  ierr = IGAGetAxis(iga,1,&axis);CHKERRQ(ierr);
  ierr = IGAAxisSetDegree(axis,p);CHKERRQ(ierr);
  ierr = IGAAxisInitUniform(axis,N/10,0.0,0.1*L,0);CHKERRQ(ierr);

  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);
  user.iga = iga;

  ierr = IGASetFormIFunction(iga,Residual,&user);CHKERRQ(ierr);
  ierr = IGASetFormIJacobian(iga,Jacobian,&user);CHKERRQ(ierr);

  ierr = IGASetBoundaryForm(iga,0,0,PETSC_TRUE);CHKERRQ(ierr);
  ierr = IGASetBoundaryForm(iga,0,0,PETSC_TRUE);CHKERRQ(ierr);

  ierr = IGASetBoundaryValue(iga,0,1,0,10.0);CHKERRQ(ierr);
  ierr = IGASetBoundaryValue(iga,0,1,1, 0.0);CHKERRQ(ierr);

  TS     ts;
  ierr = IGACreateTS(iga,&ts);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,1.0e6);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,10.0);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSALPHA);CHKERRQ(ierr);
  ierr = TSAlphaSetRadius(ts,0.95);CHKERRQ(ierr);
  ierr = TSAlphaUseAdapt(ts,PETSC_TRUE);CHKERRQ(ierr);
  ierr = TSMonitorSet(ts,Monitor,&user,NULL);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  Vec       U;
  ierr = IGACreateVec(iga,&U);CHKERRQ(ierr);
  ierr = InitialCondition(iga,U,&user);CHKERRQ(ierr);
  ierr = TSSolve(ts,U);CHKERRQ(ierr);

  ierr = IGAWrite(iga,"iga.dat");CHKERRQ(ierr);
  ierr = IGAWriteVec(iga,U,"ss.dat");CHKERRQ(ierr);

  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
