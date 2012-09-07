#include "petiga.h"

typedef struct { 
  IGA iga;
  PetscReal theta,cbar,alpha;
  PetscReal L0,lambda,tau;
  Vec X0,E;
  PetscScalar energy[3],time[3];
} AppCtx;

#undef  __FUNCT__
#define __FUNCT__ "Mobility"
void Mobility(AppCtx *user,PetscReal c,PetscReal *M,PetscReal *dM,PetscReal *d2M)
{
  if (M)   *M   = c*(1-c);
  if (dM)  *dM  = 1-2*c;
  if (d2M) *d2M = -2;
}

#undef __FUNCT__
#define __FUNCT__ "GinzburgLandauFreeEnergy"
PetscScalar GinzburgLandauFreeEnergy(PetscReal c,PetscReal cx,PetscReal cy,PetscReal cz,AppCtx *user)
{
  PetscReal E,omc = 1.0-c;
  E=c*log(c)+omc*log(omc)+2.0*user->theta*c*omc+user->theta/3.0/user->alpha*fabs(cx*cx+cy*cy+cz*cz);
  return E;
}

#undef  __FUNCT__
#define __FUNCT__ "Stats"
PetscErrorCode Stats(IGAPoint p,const PetscScalar *U,PetscInt n,PetscScalar *S,void *ctx)
{
  PetscFunctionBegin;
  AppCtx *user = (AppCtx *)ctx;
 
  PetscScalar c,c1[3];
  IGAPointFormValue(p,U,&c); 
  IGAPointFormGrad(p,U,&c1[0]);

  S[0] = GinzburgLandauFreeEnergy(c,c1[0],c1[1],c1[2],user); 
  
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "ChemicalPotential"
void ChemicalPotential(AppCtx *user,PetscReal c,PetscReal *mu,PetscReal *dmu,PetscReal *d2mu)
{
  if (mu) {
    (*mu)  = 0.5/user->theta*log(c/(1-c))+1-2*c;
    (*mu) *= user->L0*user->L0/user->lambda;
  }
  if (dmu) {
    (*dmu)  = 0.5/user->theta*1.0/(c*(1-c)) - 2;
    (*dmu) *= user->L0*user->L0/user->lambda;
  }
  if (d2mu) {
    (*d2mu)  = -0.5/user->theta*(1-2*c)/(c*c*(1-c)*(1-c));
    (*d2mu) *= user->L0*user->L0/user->lambda;
  }
}

#undef  __FUNCT__
#define __FUNCT__ "Residual"
PetscErrorCode Residual(IGAPoint p,PetscReal dt,
			PetscReal shift,const PetscScalar *V,
                        PetscReal t,const PetscScalar *U,
                        PetscScalar *R,void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;

  PetscInt nen;
  IGAPointGetSizes(p,0,&nen,0);

  PetscScalar c,c_t;
  IGAPointFormValue(p,V,&c_t);
  IGAPointFormValue(p,U,&c);

  PetscReal M,dM;
  Mobility(user,c,&M,&dM,NULL);
  PetscReal dmu;
  ChemicalPotential(user,c,NULL,&dmu,NULL);

  PetscScalar c1[3],c2[3][3];
  IGAPointFormGrad(p,U,&c1[0]);
  IGAPointFormHess(p,U,&c2[0][0]);
  PetscScalar c_x  = c1[0],    c_y  = c1[1],    c_z  = c1[2];
  PetscScalar c_xx = c2[0][0], c_yy = c2[1][1], c_zz = c2[2][2];

  const PetscReal *N0,(*N1)[3],(*N2)[3][3];
  IGAPointGetShapeFuns(p,0,(const PetscReal**)&N0);
  IGAPointGetShapeFuns(p,1,(const PetscReal**)&N1);
  IGAPointGetShapeFuns(p,2,(const PetscReal**)&N2);

  PetscInt a;
  for (a=0; a<nen; a++) {
    PetscReal Na    = N0[a];
    PetscReal Na_x  = N1[a][0];
    PetscReal Na_y  = N1[a][1];
    PetscReal Na_z  = N1[a][2];
    PetscReal Na_xx = N2[a][0][0];
    PetscReal Na_yy = N2[a][1][1];
    PetscReal Na_zz = N2[a][2][2];
    /* ----- */
    PetscScalar Ra  = 0;
    // Na * c_t
    Ra += Na * c_t; 
    // grad(Na) . ((M*dmu + dM*del2(c))) grad(C)
    PetscScalar t1 = M*dmu + dM*(c_xx+c_yy+c_zz);
    Ra += Na_x * t1 * c_x;
    Ra += Na_y * t1 * c_y;
    Ra += Na_z * t1 * c_z;
    // del2(Na) * M * del2(c)
    Ra += (Na_xx+Na_yy+Na_zz) * M * (c_xx+c_yy+c_zz);
    /* ----- */
    R[a] = Ra;
  }
  return 0;
}

#undef  __FUNCT__
#define __FUNCT__ "Tangent"
PetscErrorCode Tangent(IGAPoint p,PetscReal dt,
		       PetscReal shift,const PetscScalar *V,
		       PetscReal t,const PetscScalar *U,
                       PetscScalar *K,void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;

  PetscInt nen;
  IGAPointGetSizes(p,0,&nen,0);

  PetscScalar c_t,c;
  IGAPointFormValue(p,V,&c_t);
  IGAPointFormValue(p,U,&c);

  PetscReal M,dM,d2M;
  Mobility(user,c,&M,&dM,&d2M);
  PetscReal dmu,d2mu;
  ChemicalPotential(user,c,NULL,&dmu,&d2mu);

  PetscScalar c1[3],c2[3][3];
  IGAPointFormGrad(p,U,&c1[0]);
  IGAPointFormHess(p,U,&c2[0][0]);
  PetscScalar c_x  = c1[0],    c_y  = c1[1],    c_z  = c1[2];
  PetscScalar c_xx = c2[0][0], c_yy = c2[1][1], c_zz = c2[2][2];

  const PetscReal *N0,(*N1)[3],(*N2)[3][3];
  IGAPointGetShapeFuns(p,0,(const PetscReal**)&N0);
  IGAPointGetShapeFuns(p,1,(const PetscReal**)&N1);
  IGAPointGetShapeFuns(p,2,(const PetscReal**)&N2);

  PetscInt a,b;
  for (a=0; a<nen; a++) {
    PetscReal Na    = N0[a];
    PetscReal Na_x  = N1[a][0];
    PetscReal Na_y  = N1[a][1];
    PetscReal Na_z  = N1[a][2];
    PetscReal Na_xx = N2[a][0][0];
    PetscReal Na_yy = N2[a][1][1];
    PetscReal Na_zz = N2[a][2][2];
    for (b=0; b<nen; b++) {
      PetscReal Nb    = N0[b];
      PetscReal Nb_x  = N1[b][0];
      PetscReal Nb_y  = N1[b][1];
      PetscReal Nb_z  = N1[b][2];
      PetscReal Nb_xx = N2[b][0][0];
      PetscReal Nb_yy = N2[b][1][1];
      PetscReal Nb_zz = N2[b][2][2];
      /* ----- */
      PetscScalar Kab = 0;
      // shift*Na*Nb
      Kab += shift*Na*Nb;
      // grad(Na) . (M*dmu+dM*del2(c)) grad(Nb)
      PetscScalar t1 = M*dmu + dM*(c_xx+c_yy+c_zz);
      Kab += Na_x * t1 * Nb_x;
      Kab += Na_y * t1 * Nb_y;
      Kab += Na_z * t1 * Nb_z;
      // grad(Na) . ((dM*dmu+M*d2mu+d2M*del2(c))*Nb + dM*del2(Nb)) grad(C)
      PetscScalar t2 = (dM*dmu+M*d2mu+d2M*(c_xx+c_yy+c_zz))*Nb + dM*(Nb_xx+Nb_yy+Nb_zz);
      Kab += Na_x * t2 * c_x;
      Kab += Na_y * t2 * c_y;
      Kab += Na_z * t2 * c_z;
      // del2(Na) * ((dM*del2(c)*Nb + M*del2(Nb))
      Kab += (Na_xx+Na_yy+Na_zz) * (dM*(c_xx+c_yy+c_zz)*Nb + M*(Nb_xx+Nb_yy+Nb_zz));
      /* ----- */
      K[a*nen+b] = Kab;
    }
  }
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "FormInitialCondition"
PetscErrorCode FormInitialCondition(AppCtx *user,IGA iga,const char datafile[],Vec C)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (datafile[0] != 0) { /* initial condition from datafile */
    MPI_Comm comm;
    PetscViewer viewer;
    ierr = PetscObjectGetComm((PetscObject)C,&comm);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(comm,datafile,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = VecLoad(C,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);
  } else { /* initial condition is random */
    PetscRandom rctx;    
    ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
    ierr = PetscRandomSetInterval(rctx,user->cbar-0.05,user->cbar+0.05);CHKERRQ(ierr); 
    ierr = PetscRandomSeed(rctx);CHKERRQ(ierr);
    ierr = VecSetRandom(C,rctx);CHKERRQ(ierr); 
    ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr); 
  }
  PetscFunctionReturn(0); 
}

#undef __FUNCT__
#define __FUNCT__ "WriteSolution"
PetscErrorCode WriteSolution(Vec C, const char pattern[],int number)
{
  PetscFunctionBegin;
  PetscErrorCode  ierr;
  MPI_Comm        comm;
  char            filename[256];
  PetscViewer     viewer;

  PetscFunctionBegin;
  sprintf(filename,pattern,number);
  ierr = PetscObjectGetComm((PetscObject)C,&comm);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(comm,filename,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(C,viewer);CHKERRQ(ierr);
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "OutputMonitor"
PetscErrorCode OutputMonitor(TS ts,PetscInt step,PetscReal t,Vec U,void *mctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = WriteSolution(U,"ch%d.dat",step);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StatsMonitor"
PetscErrorCode StatsMonitor(TS ts,PetscInt step,PetscReal t,Vec U,void *mctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  AppCtx *user = (AppCtx *)mctx;
  if (!user->E) PetscFunctionReturn(0);

  PetscScalar dt;
  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"%.6e %.6e %.16e\n",t,dt,user->energy[0]);
  
  if(user->energy[0] > user->energy[1]) PetscPrintf(PETSC_COMM_WORLD,"WARNING: Ginzburg-Landau free energy increased!\n");

  PetscFunctionReturn(0);
}

PetscScalar EstimateSecondDerivative(AppCtx user)
{
  PetscScalar dE = user.time[0]*(user.energy[2]-user.energy[1]);
  dE += user.time[1]*(user.energy[0]-user.energy[2]);
  dE += user.time[2]*(user.energy[1]-user.energy[0]);
  dE /= (user.time[0]*(user.time[0]*user.time[1]-user.time[0]*user.time[2]-user.time[1]*user.time[1]+user.time[2]*user.time[2]) + user.time[1]*user.time[1]*user.time[2] - user.time[1]*user.time[2]*user.time[2]);
  return dE;
}

#undef __FUNCT__
#define __FUNCT__ "CHAdapt"
PetscErrorCode CHAdapt(TS ts,PetscReal t,Vec X,Vec Xdot, PetscReal *nextdt,PetscBool *ok,void *ctx)
{
  SNES                 snes;
  SNESConvergedReason  snesreason;
  PetscReal            dt,normE,normX,Emax,scale;
  PetscErrorCode       ierr;
  PetscFunctionBegin;

  AppCtx *th = (AppCtx *)ctx;

  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  th->time[0]=t-dt;
  ierr = IGAFormScalar(th->iga,X,1,&th->energy[0],Stats,ctx);CHKERRQ(ierr);

  if (!th->E){
    th->energy[1] = 1.01*th->energy[0];
    th->energy[2] = 1.02*th->energy[0];
    th->time[1] = -dt;
    th->time[2] = -2.0*dt;
  }

  PetscScalar scale_min = 0.5;
  PetscScalar scale_max = 1.5;
  PetscScalar    dt_min = 1.0e-15;
  PetscScalar    dt_max = 1.0e+10;
  PetscScalar      rtol = 1.0e-3;
  PetscScalar      atol = 1.0e-3;
  PetscScalar       rho = 0.9;
  //PetscScalar       dE2 = EstimateSecondDerivative(*th);
  
  /* If the SNES fails, reject the step and reduce the step by the
     maximum amount */
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESGetConvergedReason(snes,&snesreason);CHKERRQ(ierr);
  if (snesreason < 0 || th->energy[0] > th->energy[1]) {
    *ok = PETSC_FALSE;
    *nextdt *= scale_min;
    *nextdt = PetscMax(*nextdt,dt_min);
    PetscFunctionReturn(0);
  }

  /* first-order aproximation to the local error */
  /* E = (X0 + dt*Xdot) - X */
  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  if (!th->E) {ierr = VecDuplicate(th->X0,&th->E);CHKERRQ(ierr);}
  ierr = VecWAXPY(th->E,dt,Xdot,th->X0);CHKERRQ(ierr);
  ierr = VecAXPY(th->E,-1,X);CHKERRQ(ierr);
  ierr = VecNorm(th->E,NORM_2,&normE);CHKERRQ(ierr);
  /* compute maximum allowable error */
  ierr = VecNorm(X,NORM_2,&normX);CHKERRQ(ierr);
  if (normX == 0) {ierr = VecNorm(th->X0,NORM_2,&normX);CHKERRQ(ierr);}
  Emax =  rtol * normX + atol;
  /* compute next time step */
  if (normE > 0) {
    scale = rho * PetscRealPart(PetscSqrtScalar((PetscScalar)(Emax/normE)));
    scale = PetscMax(scale,scale_min);
    scale = PetscMin(scale,scale_max);
    if (!(*ok))
      scale = PetscMin(1.0,scale);
    *nextdt *= scale;
  }
  /* accept or reject step */
  if (normE <= Emax){
    ierr = VecCopy(X,th->X0);CHKERRQ(ierr);
    th->energy[2] = th->energy[1];
    th->energy[1] = th->energy[0];
    th->time[2] = th->time[1];
    th->time[1] = th->time[0];
    *ok = PETSC_TRUE;
  }else{
    *ok = PETSC_FALSE;
  }

  *nextdt = PetscMax(*nextdt,dt_min);
  *nextdt = PetscMin(*nextdt,dt_max);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[]) {

  PetscErrorCode  ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  /* Define simulation specific parameters */
  AppCtx user;
  user.cbar  = 0.63;   /* average concentration */
  user.alpha = 200.0;  /* thickess interface parameter */
  user.theta = 1.5;    /* temperature/critical temperature */
  user.L0    = 1.0;    /* length scale */
  user.tau   = 1.0;

  PetscBool output = PETSC_FALSE; 
  PetscBool monitor = PETSC_FALSE; 
  char initial[PETSC_MAX_PATH_LEN] = {0};
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","CahnHilliard3D Options","CH3D");CHKERRQ(ierr);
  ierr = PetscOptionsString("-ch_initial","Load initial solution from file",__FILE__,initial,initial,sizeof(initial),PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ch_output","Enable output files",__FILE__,output,&output,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ch_monitor","Compute and show statistics of solution",__FILE__,monitor,&monitor,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ch_cbar","Initial average concentration",__FILE__,user.cbar,&user.cbar,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ch_alpha","Characteristic parameter",__FILE__,user.alpha,&user.alpha,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDim(iga,3);CHKERRQ(ierr);
  ierr = IGASetDof(iga,1);CHKERRQ(ierr);
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  PetscReal h = 1.0/sqrt(iga->elem_sizes[0]*iga->elem_sizes[0]+
			 iga->elem_sizes[1]*iga->elem_sizes[1]+
			 iga->elem_sizes[2]*iga->elem_sizes[2]);
  user.lambda = user.tau*h*h; /* mesh size parameter */

  ierr = IGASetUserIFunction(iga,Residual,&user);CHKERRQ(ierr);
  ierr = IGASetUserIJacobian(iga,Tangent,&user);CHKERRQ(ierr);

  TS ts;
  ierr = IGACreateTS(iga,&ts);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,10000,1.0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,1e-10);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSALPHA);CHKERRQ(ierr);
  ierr = TSAlphaSetRadius(ts,0.5);CHKERRQ(ierr);

  user.E = PETSC_NULL;
  ierr = TSAlphaSetAdapt(ts,CHAdapt,&user);CHKERRQ(ierr); 

  if (monitor) {
    user.iga = iga;
    PetscPrintf(PETSC_COMM_WORLD,"#Time        dt           Free Energy            Second moment          Third moment\n");
    ierr = TSMonitorSet(ts,StatsMonitor,&user,PETSC_NULL);CHKERRQ(ierr);
  }
  if (output) {
    ierr = TSMonitorSet(ts,OutputMonitor,&user,PETSC_NULL);CHKERRQ(ierr);
  }

  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  PetscReal t; Vec U;
  ierr = IGACreateVec(iga,&U);CHKERRQ(ierr);
  ierr = FormInitialCondition(&user,iga,initial,U);CHKERRQ(ierr);
  ierr = VecDuplicate(U,&user.X0);CHKERRQ(ierr);
  ierr = VecCopy(U,user.X0);CHKERRQ(ierr);
  ierr = TSSolve(ts,U,&t);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

