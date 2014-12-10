#include "petiga.h"

#if PETSC_VERSION_LE(3,3,0)
#define TSSolve(ts,x) TSSolve((ts),(x),NULL)
#endif

typedef struct { 
  PetscReal theta,cbar,alpha;
  PetscReal Eprev;
} AppCtx;

#undef  __FUNCT__
#define __FUNCT__ "Mobility"
void Mobility(AppCtx *user,PetscReal c,PetscReal *M,PetscReal *dM,PetscReal *d2M)
{
  if (M)   *M   = c*(1-c);
  if (dM)  *dM  = 1-2*c;
  if (d2M) *d2M = -2;
}

#undef  __FUNCT__
#define __FUNCT__ "ChemicalPotential"
void ChemicalPotential(AppCtx *user,PetscReal c,PetscReal *mu,PetscReal *dmu,PetscReal *d2mu)
{
  PetscReal theta  = user->theta;
  PetscReal alpha  = user->alpha;
  if (mu) {
    (*mu)  = 0.5/theta*log(c/(1-c)) + 1 - 2*c;
    (*mu) *= 3*alpha;
  }
  if (dmu) {
    (*dmu)  = 0.5/theta*1/(c*(1-c)) - 2;
    (*dmu) *= 3*alpha;
  }
  if (d2mu) {
    (*d2mu)  = 0.5/theta*(2*c-1)/(c*c*(1-c)*(1-c));
    (*d2mu) *= 3*alpha;
  }
}


#undef  __FUNCT__
#define __FUNCT__ "GinzburgLandauFreeEnergy"
PetscReal GinzburgLandauFreeEnergy(PetscReal c,PetscReal cx,PetscReal cy,AppCtx *user)
{
  PetscReal theta = user->theta;
  PetscReal alpha = user->alpha;
  PetscReal E = c*log(c) + (1-c)*log(1-c) + 2*theta*c*(1-c) + theta/(3*alpha)*(cx*cx+cy*cy);
  return E;
}

#undef   __FUNCT__
#define __FUNCT__ "Stats"
PetscErrorCode Stats(IGAPoint p,const PetscScalar *U,PetscInt n,PetscScalar *S,void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;
  PetscFunctionBegin;
 
  PetscScalar c,c1[3];
  IGAPointFormValue(p,U,&c); 
  IGAPointFormGrad(p,U,&c1[0]);
  PetscReal diff = c - user->cbar;

  S[0] = GinzburgLandauFreeEnergy(c,c1[0],c1[1],user); // Free energy
  S[1] = diff*diff;                                    // Second moment
  S[2] = S[1]*diff;                                    // Third moment
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StatsMonitor"
PetscErrorCode StatsMonitor(TS ts,PetscInt step,PetscReal t,Vec U,void *mctx)
{
  AppCtx         *user = (AppCtx *)mctx;
  IGA            iga;
  PetscReal      dt;
  PetscScalar    stats[3] = {0,0,0};
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)ts,"IGA",(PetscObject*)&iga);CHKERRQ(ierr);
  PetscValidHeaderSpecific(iga,IGA_CLASSID,0);

  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  ierr = IGAComputeScalar(iga,U,3,&stats[0],Stats,mctx);CHKERRQ(ierr);

  if (step == 0) {ierr = PetscPrintf(PETSC_COMM_WORLD,"#Time        dt           Free Energy            Second moment          Third moment\n");CHKERRQ(ierr);}
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%.6e %.6e %.16e %.16e %.16e\n",(double)t,(double)dt,(double)stats[0],(double)stats[1],(double)stats[2]);CHKERRQ(ierr);

  if (step == 0) user->Eprev = PETSC_MAX_REAL;
  if((PetscReal)stats[0] > user->Eprev) {ierr = PetscPrintf(PETSC_COMM_WORLD,"WARNING: Ginzburg-Landau free energy increased!\n");CHKERRQ(ierr);}
  user->Eprev = PetscRealPart(stats[0]);
  PetscFunctionReturn(0);
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

  PetscScalar c_t,c;
  IGAPointFormValue(p,V,&c_t);
  IGAPointFormValue(p,U,&c);

  PetscReal M,dM;
  Mobility(user,c,&M,&dM,NULL);
  PetscReal dmu;
  ChemicalPotential(user,c,NULL,&dmu,NULL);

  PetscScalar c1[2],c2[2][2];
  IGAPointFormGrad(p,U,&c1[0]);
  IGAPointFormHess(p,U,&c2[0][0]);
  PetscScalar c_x  = c1[0],    c_y  = c1[1];
  PetscScalar c_xx = c2[0][0], c_yy = c2[1][1];

  const PetscReal *N0,(*N1)[2],(*N2)[2][2];
  IGAPointGetShapeFuns(p,0,(const PetscReal**)&N0);
  IGAPointGetShapeFuns(p,1,(const PetscReal**)&N1);
  IGAPointGetShapeFuns(p,2,(const PetscReal**)&N2);

  PetscInt a;
  for (a=0; a<nen; a++) {
    PetscReal Na    = N0[a];
    PetscReal Na_x  = N1[a][0];
    PetscReal Na_y  = N1[a][1];
    PetscReal Na_xx = N2[a][0][0];
    PetscReal Na_yy = N2[a][1][1];
    /* ----- */
    PetscScalar Ra  = 0;
    // Na * c_t
    Ra += Na * c_t; 
    // grad(Na) . ((M*dmu + dM*del2(c))) grad(C)
    PetscScalar t1 = M*dmu + dM*(c_xx+c_yy);
    Ra += Na_x * t1 * c_x;
    Ra += Na_y * t1 * c_y;
    // del2(Na) * M * del2(c)
    Ra += (Na_xx+Na_yy) * M * (c_xx+c_yy);
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

  PetscScalar c1[2],c2[2][2];
  IGAPointFormGrad(p,U,&c1[0]);
  IGAPointFormHess(p,U,&c2[0][0]);
  PetscScalar c_x  = c1[0],    c_y  = c1[1];
  PetscScalar c_xx = c2[0][0], c_yy = c2[1][1];

  const PetscReal *N0,(*N1)[2],(*N2)[2][2];
  IGAPointGetShapeFuns(p,0,(const PetscReal**)&N0);
  IGAPointGetShapeFuns(p,1,(const PetscReal**)&N1);
  IGAPointGetShapeFuns(p,2,(const PetscReal**)&N2);

  PetscInt a,b;
  for (a=0; a<nen; a++) {
    PetscReal Na    = N0[a];
    PetscReal Na_x  = N1[a][0];
    PetscReal Na_y  = N1[a][1];
    PetscReal Na_xx = N2[a][0][0];
    PetscReal Na_yy = N2[a][1][1];
    for (b=0; b<nen; b++) {
      PetscReal Nb    = N0[b];
      PetscReal Nb_x  = N1[b][0];
      PetscReal Nb_y  = N1[b][1];
      PetscReal Nb_xx = N2[b][0][0];
      PetscReal Nb_yy = N2[b][1][1];
      /* ----- */
      PetscScalar Kab = 0;
      // shift*Na*Nb
      Kab += shift*Na*Nb;
      // grad(Na) . (M*dmu+dM*del2(c)) grad(Nb)
      PetscScalar t1 = M*dmu + dM*(c_xx+c_yy);
      Kab += Na_x * t1 * Nb_x;
      Kab += Na_y * t1 * Nb_y;
      // grad(Na) . ((dM*dmu+M*d2mu+d2M*del2(c))*Nb + dM*del2(Nb)) grad(C)
      PetscScalar t2 = (dM*dmu+M*d2mu+d2M*(c_xx+c_yy))*Nb + dM*(Nb_xx+Nb_yy);
      Kab += Na_x * t2 * c_x;
      Kab += Na_y * t2 * c_y;
      // del2(Na) * ((dM*del2(c)*Nb + M*del2(Nb))
      Kab += (Na_xx+Na_yy) * (dM*(c_xx+c_yy)*Nb + M*(Nb_xx+Nb_yy));
      /* ----- */
      K[a*nen+b] = Kab;
    }
  }
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "FormInitialCondition"
PetscErrorCode FormInitialCondition(IGA iga,Vec C,AppCtx *user)
{
  MPI_Comm       comm;
  PetscRandom    rctx;    
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
  ierr = PetscRandomCreate(comm,&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rctx,user->cbar-0.05,user->cbar+0.05);CHKERRQ(ierr);
  ierr = PetscRandomSeed(rctx);CHKERRQ(ierr);
  ierr = VecSetRandom(C,rctx);CHKERRQ(ierr); 
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr); 
  PetscFunctionReturn(0); 
}

#undef __FUNCT__
#define __FUNCT__ "OutputMonitor"
PetscErrorCode OutputMonitor(TS ts,PetscInt step,PetscReal t,Vec U,void *mctx)
{
  IGA            iga; 
  char           filename[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)ts,"IGA",(PetscObject*)&iga);CHKERRQ(ierr);
  PetscValidHeaderSpecific(iga,IGA_CLASSID,0);
  ierr = PetscSNPrintf(filename,sizeof(filename),"./ch2d%d.dat",(int)step);CHKERRQ(ierr);
  ierr = IGAWriteVec(iga,U,filename);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[]) {

  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  /* Define simulation specific parameters */
  AppCtx user;
  user.cbar  = 0.63;   /* average concentration */
  user.alpha = 3000.0; /* interface thickess parameter */
  user.theta = 1.5;    /* temperature/critical temperature */

  /* Set discretization options */
  PetscInt  N = 64;
  PetscInt  p = 2;
  PetscInt  k = PETSC_DECIDE;
  char      initial[PETSC_MAX_PATH_LEN] = {0};
  PetscBool output  = PETSC_FALSE; 
  PetscBool monitor = PETSC_FALSE; 
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","CahnHilliard2D Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-N","number of elements (along one dimension)",__FILE__,N,&N,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-p","polynomial order",__FILE__,p,&p,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-k","global continuity order",__FILE__,k,&k,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-initial","Load initial solution from file",__FILE__,initial,initial,sizeof(initial),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-output","Enable output files",__FILE__,output,&output,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-monitor","Compute and show statistics of solution",__FILE__,monitor,&monitor,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-cbar","Initial average concentration",__FILE__,user.cbar,&user.cbar,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-alpha","Interface thickess parameter",__FILE__,user.alpha,&user.alpha,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-theta","Ratio temperature/critical temperature",__FILE__,user.alpha,&user.alpha,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (k == PETSC_DECIDE) k = p-1;

  if (p < 2 || k < 1) /* Problem requires a p>=2 C^1 basis */
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,
            "Problem requires minimum of p = 2 and k = 1");
  if (p <= k)         /* Check k < p */
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
  ierr = IGAAxisInitUniform(axis0,N,0.0,1.0,k);CHKERRQ(ierr);
  IGAAxis axis1;
  ierr = IGAGetAxis(iga,1,&axis1);CHKERRQ(ierr);
  ierr = IGAAxisCopy(axis0,axis1);CHKERRQ(ierr);

  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  ierr = IGASetFormIFunction(iga,Residual,&user);CHKERRQ(ierr);
  ierr = IGASetFormIJacobian(iga,Tangent,&user);CHKERRQ(ierr);

  TS ts;
  ierr = IGACreateTS(iga,&ts);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,10000,1.0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,1e-11);CHKERRQ(ierr);

  ierr = TSSetType(ts,TSALPHA1);CHKERRQ(ierr);
  ierr = TSAlphaSetRadius(ts,0.5);CHKERRQ(ierr);
  ierr = TSAlphaUseAdapt(ts,PETSC_TRUE);CHKERRQ(ierr);
  ierr = TSSetMaxSNESFailures(ts,-1);CHKERRQ(ierr);

  if (monitor) {ierr = TSMonitorSet(ts,StatsMonitor,&user,NULL);CHKERRQ(ierr);}
  if (output)  {ierr = TSMonitorSet(ts,OutputMonitor,&user,NULL);CHKERRQ(ierr);}
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  Vec C;
  ierr = TSGetSolution(ts,&C);CHKERRQ(ierr);
  if (initial[0] != 0) { /* initial condition from datafile */
    ierr = IGAReadVec(iga,C,initial);CHKERRQ(ierr);
  } else {                /* initial condition is random */
    ierr = FormInitialCondition(iga,C,&user);CHKERRQ(ierr);
  }
  ierr = TSSolve(ts,C);CHKERRQ(ierr);

  ierr = VecDestroy(&C);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
