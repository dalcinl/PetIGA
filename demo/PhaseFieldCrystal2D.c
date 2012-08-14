#include "petiga.h"

typedef struct { 
  IGA iga;
  PetscReal cbar,D,k,Eps,g;
  PetscReal L0,Sprev[3];
} AppCtx;

#undef __FUNCT__
#define __FUNCT__ "GinzburgLandauFreeEnergy"
PetscScalar GinzburgLandauFreeEnergy(PetscReal c,PetscReal cx,PetscReal cy,AppCtx *user)
{
  PetscReal E;
  E=-user->Eps/2.0*c*c-user->g/3.0*c*c*c+c*c*c/4.0+user->D/2.0*((cx*cx+cy*cy)*(cx*cx+cy*cy)-2.0*user->k*user->k*fabs(cx*cx+cy*cy)+user->k*user->k*user->k*user->k*c*c);
  return E;
}

#undef  __FUNCT__
#define __FUNCT__ "Stats"
PetscErrorCode Stats(IGAPoint p,const PetscScalar *U,PetscInt n,PetscScalar *S,void *ctx)
{
  PetscFunctionBegin;
  AppCtx *user = (AppCtx *)ctx;
 
  PetscScalar c,c1[3];
  IGAPointGetValue(p,U,&c); 
  IGAPointGetGrad(p,U,&c1[0]);
  PetscScalar diff = c - user->cbar;

  S[0] = GinzburgLandauFreeEnergy(c,c1[0],c1[1],user); // Free energy
  S[1] = diff*diff;                                    // Second moment
  S[2] = S[1]*diff;                                    // Third moment
  
  PetscFunctionReturn(0);
}

//#undef  __FUNCT__
//#define __FUNCT__ "Mobility"
//void Mobility(AppCtx *user,PetscReal c,PetscReal *M,PetscReal *dM,PetscReal *d2M)
//{
//  if (M)   *M   = c*(1-c);
//  if (dM)  *dM  = 1-2*c;
//  if (d2M) *d2M = -2;
//}

#undef __FUNCT__
#define __FUNCT__ "StatsMonitor"
PetscErrorCode StatsMonitor(TS ts,PetscInt step,PetscReal t,Vec U,void *mctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  AppCtx *user = (AppCtx *)mctx;

  PetscScalar stats[3] = {0,0,0};
  ierr = IGAFormScalar(user->iga,U,3,&stats[0],Stats,mctx);CHKERRQ(ierr);

  PetscScalar dt;
  TSGetTimeStep(ts,&dt);
  PetscPrintf(PETSC_COMM_WORLD,"%.6e %.6e %.16e %.16e %.16e\n",t,dt,stats[0],stats[1],stats[2]);

  if(stats[0] > user->Sprev[0]) PetscPrintf(PETSC_COMM_WORLD,"WARNING: Ginzburg-Landau free energy increased!\n");
  user->Sprev[0] = stats[0];

  PetscFunctionReturn(0);
}

//#undef  __FUNCT__
//#define __FUNCT__ "ChemicalPotential"
//void ChemicalPotential(AppCtx *user,PetscReal c,PetscReal *mu,PetscReal *dmu,PetscReal *d2mu)
//{
//  if (mu) {
//   (*mu)  = 0.5/user->theta*log(c/(1-c))+1-2*c;
//    (*mu) *= user->L0*user->L0/user->lambda;
//  }
//  if (dmu) {
//    (*dmu)  = 0.5/user->theta*1.0/(c*(1-c)) - 2;
//    (*dmu) *= user->L0*user->L0/user->lambda;
//  }
//  if (d2mu) {
//    (*d2mu)  = -0.5/user->theta*(1-2*c)/(c*c*(1-c)*(1-c));
//    (*d2mu) *= user->L0*user->L0/user->lambda;
//  }
//}

#undef  __FUNCT__
#define __FUNCT__ "Residual"
PetscErrorCode Residual(IGAPoint p,PetscReal dt,
                        PetscReal shift,const PetscScalar *V,
                        PetscReal t,const PetscScalar *U,
                        PetscScalar *R,void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;

  PetscInt nen;
  IGAPointGetSizes(p,&nen,0,0);

  PetscScalar c_t,c;
  IGAPointGetValue(p,V,&c_t);
  IGAPointGetValue(p,U,&c);

  //PetscReal M,dM;
  //Mobility(user,c,&M,&dM,NULL);
  //PetscReal dmu;
  //ChemicalPotential(user,c,NULL,&dmu,NULL);

  PetscScalar c1[2],c2[2][2],c3[2][2][2];
  IGAPointGetGrad(p,U,&c1[0]);
  IGAPointGetHess(p,U,&c2[0][0]);
  IGAPointGetDer3(p,U,&c3[0][0][0]);
  PetscScalar c_x   = c1[0],       c_y   = c1[1];
  PetscScalar c_xx  = c2[0][0],    c_yy  = c2[1][1];
  PetscScalar c_xxx = c3[0][0][0], c_yyy = c3[1][1][1];
  PetscScalar c_xxy = c3[0][0][1], c_xyy = c3[0][1][1];

  const PetscReal *N0,(*N1)[2],(*N2)[2][2],(*N3)[2][2][2];
  IGAPointGetShapeFuns(p,0,(const PetscReal**)&N0);
  IGAPointGetShapeFuns(p,1,(const PetscReal**)&N1);
  IGAPointGetShapeFuns(p,2,(const PetscReal**)&N2);
  IGAPointGetShapeFuns(p,3,(const PetscReal**)&N3);

  PetscInt a;
  for (a=0; a<nen; a++) {
    PetscReal Na     = N0[a];
    PetscReal Na_x   = N1[a][0];
    PetscReal Na_y   = N1[a][1];
    PetscReal Na_xx  = N2[a][0][0];
    PetscReal Na_yy  = N2[a][1][1];
    PetscReal Na_xxx = N3[a][0][0][0];
    PetscReal Na_yyy = N3[a][1][1][1];
    PetscReal Na_xxy = N3[a][0][0][1];
    PetscReal Na_xyy = N3[a][0][1][1];

    /* ----- */
    PetscScalar Ra  = 0;
    // Na * c_t
    Ra += Na * c_t; 
    // grad(Na) . ((-\epsilon\phi-g\epsilon^2+\phi^3)) grad(c)
    PetscScalar t1 = -user->Eps*c - user->g*c*c +c*c*c;
    Ra += Na_x * t1 * c_x;
    Ra += Na_y * t1 * c_y;
    // -2Dk^2 . del2(Na)  . del2(c)
    Ra += -2.0*user->D*user->k*user->k * (Na_xx+Na_yy) * (c_xx+c_yy);
    // D . del3(Na) . del3(c)
    Ra += user->D * (Na_yyy * (c_yyy + c_xxy) + Na_xxx * (c_xxx + c_xyy) + Na_xyy * (c_xxx + c_xyy) + Na_xxy * (c_yyy + c_xxy));
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
  IGAPointGetSizes(p,&nen,0,0);

  PetscScalar c_t,c;
  IGAPointGetValue(p,V,&c_t);
  IGAPointGetValue(p,U,&c);

  //PetscReal M,dM,d2M;
  //Mobility(user,c,&M,&dM,&d2M);
  // PetscReal dmu,d2mu;
  // ChemicalPotential(user,c,NULL,&dmu,&d2mu);

  PetscScalar c1[2],c2[2][2],c3[2][2][2];
  IGAPointGetGrad(p,U,&c1[0]);
  IGAPointGetHess(p,U,&c2[0][0]);
  IGAPointGetDer3(p,U,&c3[0][0][0]);
  PetscScalar c_x   = c1[0],       c_y   = c1[1];
  PetscScalar c_xx  = c2[0][0],    c_yy  = c2[1][1];
  PetscScalar c_xxx = c3[0][0][0], c_yyy = c3[1][1][1];
  PetscScalar c_xxy = c3[0][0][1], c_xyy = c3[0][1][1];

  const PetscReal *N0,(*N1)[2],(*N2)[2][2],(*N3)[2][2][2];
  IGAPointGetShapeFuns(p,0,(const PetscReal**)&N0);
  IGAPointGetShapeFuns(p,1,(const PetscReal**)&N1);
  IGAPointGetShapeFuns(p,2,(const PetscReal**)&N2);
  IGAPointGetShapeFuns(p,3,(const PetscReal**)&N3);

  PetscInt a,b;
  for (a=0; a<nen; a++) {
    PetscReal Na     = N0[a];
    PetscReal Na_x   = N1[a][0];
    PetscReal Na_y   = N1[a][1];
    PetscReal Na_xx  = N2[a][0][0];
    PetscReal Na_yy  = N2[a][1][1];
    PetscReal Na_xxx = N3[a][0][0][0];
    PetscReal Na_yyy = N3[a][1][1][1];
    PetscReal Na_xxy = N3[a][0][0][1];
    PetscReal Na_xyy = N3[a][0][1][1];

    for (b=0; b<nen; b++) {
      PetscReal Nb     = N0[b];
      PetscReal Nb_x   = N1[b][0];
      PetscReal Nb_y   = N1[b][1];
      PetscReal Nb_xx  = N2[b][0][0];
      PetscReal Nb_yy  = N2[b][1][1];
      PetscReal Nb_xxx = N3[b][0][0][0];
      PetscReal Nb_yyy = N3[b][1][1][1];
      PetscReal Nb_xxy = N3[b][0][0][1];
      PetscReal Nb_xyy = N3[b][0][1][1];

      /* ----- */
      PetscScalar Kab = 0;
      // shift*Na*Nb
      Kab += shift*Na*Nb;
      // grad(Na) . (-Eps*c-g*c^2+c^3) grad(Nb)
      PetscScalar t1 = -user->Eps*c - user->g*c*c +c*c*c;
      Kab += Na_x * t1 * Nb_x;
      Kab += Na_y * t1 * Nb_y;
      // grad(Na) . (deriv(t1)) grad(c)
      PetscScalar t2 = (-user->Eps*Nb-user->g*Nb*Nb+Nb*Nb*Nb);
      Kab += Na_x * t2 * c_x;
      Kab += Na_y * t2 * c_y;
      // -2Dk^2 * del2(Na) * del2(Nb)
      Kab += -2.0*user->D*user->k*user->k*(Na_xx+Na_yy) * (Nb_xx+Nb_yy);
      // D * del3(Na) * del3(Nb)
      Kab += user->D*(Na_yyy * (Nb_yyy + Nb_xxy) + Na_xxx * (Nb_xxx + Nb_xyy) + Na_xyy * (Nb_xxx + Nb_xyy) + Na_xxy * (Nb_yyy + Nb_xxy));
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
    ierr = PetscRandomSetInterval(rctx,user->cbar-0.000001,user->cbar+0.0000001);CHKERRQ(ierr); 
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
#define __FUNCT__ "main"
int main(int argc, char *argv[]) {

  PetscErrorCode  ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  /* Define simulation specific parameters */
  AppCtx user;
  user.cbar  = 0.285;   /* average concentration */
  user.D     = 1.0;     /* thickess interface parameter */
  user.k     = 1.0;     /* thickess interface parameter */
  user.Eps   = 0.25;    /* thickess interface parameter */
  user.g     = 0.0;     /* thickess interface parameter */
  user.L0    = 800.0;   /* length scale */
  user.Sprev[0] = user.Sprev[1] = user.Sprev[2] = 1.0e20; 

  /* Set discretization options */
  PetscInt N=64, p=2, C=PETSC_DECIDE;
  PetscBool output = PETSC_FALSE; 
  PetscBool monitor = PETSC_FALSE; 
  char initial[PETSC_MAX_PATH_LEN] = {0};
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","CahnHilliard2D Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-N","number of elements (along one dimension)",__FILE__,N,&N,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-p","polynomial order",__FILE__,p,&p,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-C","global continuity order",__FILE__,C,&C,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-ch_initial","Load initial solution from file",__FILE__,initial,initial,sizeof(initial),PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ch_output","Enable output files",__FILE__,output,&output,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ch_monitor","Compute and show statistics of solution",__FILE__,monitor,&monitor,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ch_cbar","Initial atomistic density field",__FILE__,user.cbar,&user.cbar,PETSC_NULL);CHKERRQ(ierr);
  //ierr = PetscOptionsReal("-ch_alpha","Characteristic parameter",__FILE__,user.alpha,&user.alpha,PETSC_NULL);CHKERRQ(ierr);
ierr = PetscOptionsReal("-ch_g","Physical parameter",__FILE__,user.g,&user.g,PETSC_NULL);CHKERRQ(ierr);
ierr = PetscOptionsReal("-ch_k","Positive number",__FILE__,user.k,&user.k,PETSC_NULL);CHKERRQ(ierr);
ierr = PetscOptionsReal("-ch_Eps","Physical parameter",__FILE__,user.Eps,&user.Eps,PETSC_NULL);CHKERRQ(ierr);
ierr = PetscOptionsReal("-ch_D","Positive number",__FILE__,user.D,&user.D,PETSC_NULL);CHKERRQ(ierr);



  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (C == PETSC_DECIDE) C = p-1;

  // user.lambda = 1.0/N/N; /* mesh size parameter */
  
  if (p < 2 || C < 1) /* Problem requires a p>=2 C1 basis */
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,
            "Problem requires minimum of p = 2 and C = 1");
  if (p <= C)         /* Check C < p */
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,
            "Discretization inconsistent: polynomial order must be greater than degree of continuity");

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDim(iga,2);CHKERRQ(ierr);
  ierr = IGASetDof(iga,1);CHKERRQ(ierr);

  IGAAxis axis0;
  ierr = IGAGetAxis(iga,0,&axis0);CHKERRQ(ierr);
  ierr = IGAAxisSetPeriodic(axis0,PETSC_TRUE);CHKERRQ(ierr);
  ierr = IGAAxisSetDegree(axis0,p);CHKERRQ(ierr);
  ierr = IGAAxisInitUniform(axis0,N,0.0,user.L0,C);CHKERRQ(ierr);
  IGAAxis axis1;
  ierr = IGAGetAxis(iga,1,&axis1);CHKERRQ(ierr);
  ierr = IGAAxisCopy(axis0,axis1);CHKERRQ(ierr);

  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  ierr = IGASetUserIFunction(iga,Residual,&user);CHKERRQ(ierr);
  ierr = IGASetUserIJacobian(iga,Tangent,&user);CHKERRQ(ierr);


  TS ts;
  ierr = IGACreateTS(iga,&ts);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,10000000,2000.0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,1e-1);CHKERRQ(ierr);

  ierr = TSSetType(ts,TSALPHA);CHKERRQ(ierr);
  ierr = TSAlphaSetRadius(ts,1.0);CHKERRQ(ierr);
  //ierr = TSAlphaSetAdapt(ts,TSAlphaAdaptDefault,PETSC_NULL);CHKERRQ(ierr); 

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
  ierr = TSSolve(ts,U,&t);CHKERRQ(ierr);

  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
