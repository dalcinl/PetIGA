#include "petiga.h"

typedef struct { 
  IGA iga;
  PetscReal cbar,D,k,Eps,g;
  PetscReal L0,Sprev[3];
  PetscReal C1[2],C2[2],C3[2];
  PetscReal ang[3];
  PetscReal dist;
  PetscReal coefC, coefq;
} AppCtx;

#define SQ(A) ((A)*(A))

#undef __FUNCT__
#define __FUNCT__ "FreeEnergy"
PetscScalar FreeEnergy(PetscReal c,PetscReal cx,PetscReal cy,PetscReal cxx,PetscReal cyy,AppCtx *user)
{
  PetscReal E;
  E  = -user->Eps/2.0*c*c;
  E += -user->g/3.0*c*c*c;
  E +=  0.25*c*c*c*c;
  E +=  0.5*user->D* ( SQ(cxx*cxx+cyy*cyy) - 2.0*SQ(user->k)*fabs(cx*cx+cy*cy) + SQ(user->k)*SQ(user->k)*c*c );
  return E;
}

#undef  __FUNCT__
#define __FUNCT__ "Stats"
PetscErrorCode Stats(IGAPoint p,const PetscScalar *U,PetscInt n,PetscScalar *S,void *ctx)
{
  PetscFunctionBegin;
  AppCtx *user = (AppCtx *)ctx;
 
  PetscScalar c,c1[2],c2[2][2];
  IGAPointFormValue(p,U,&c); 
  IGAPointFormGrad(p,U,&c1[0]);
  IGAPointFormHess(p,U,&c2[0][0]);

  S[0] = FreeEnergy(c,c1[0],c1[1],c2[0][0],c2[1][1],user); // Free energy
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StatsMonitor"
PetscErrorCode StatsMonitor(TS ts,PetscInt step,PetscReal t,Vec U,void *mctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  AppCtx *user = (AppCtx *)mctx;

  PetscScalar stats[1] = {0};
  ierr = IGAFormScalar(user->iga,U,1,&stats[0],Stats,mctx);CHKERRQ(ierr);

  PetscScalar dt;
  TSGetTimeStep(ts,&dt);
  PetscPrintf(PETSC_COMM_WORLD,"%.6e %.6e %.16e\n",t,dt,stats[0]);

  if(stats[0] > user->Sprev[0]) PetscPrintf(PETSC_COMM_WORLD,"WARNING: free energy increased!\n");
  user->Sprev[0] = stats[0];

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

  PetscScalar c1[2],c2[2][2],c3[2][2][2];
  IGAPointFormGrad(p,U,&c1[0]);
  IGAPointFormHess(p,U,&c2[0][0]);
  IGAPointFormDer3(p,U,&c3[0][0][0]);
  PetscScalar c_x   = c1[0],       c_y   = c1[1];
  PetscScalar c_xx  = c2[0][0],    c_yy  = c2[1][1];
  PetscScalar c_xxx = c3[0][0][0], c_yyy = c3[1][1][1];
  PetscScalar c_xxy = c3[0][0][1], c_xyy = c3[0][1][1];

  const PetscReal *N0,(*N1)[2],(*N2)[2][2],(*N3)[2][2][2];
  IGAPointGetShapeFuns(p,0,(const PetscReal**)&N0);
  IGAPointGetShapeFuns(p,1,(const PetscReal**)&N1);
  IGAPointGetShapeFuns(p,2,(const PetscReal**)&N2);
  IGAPointGetShapeFuns(p,3,(const PetscReal**)&N3);

  PetscReal k4,k2 = user->k*user->k; k4 = k2*k2;
  PetscReal D = user->D, eps = user->Eps, g = user->g;
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
    // grad(Na) . ((-\epsilon-2g\epsilon+3\phi^2 +Dk^4)) grad(c)
    PetscScalar t1 = -eps - 2.0*g*c + 3.0*c*c + D*k4;
    Ra += Na_x * t1 * c_x;
    Ra += Na_y * t1 * c_y;
    // -2Dk^2 . del2(Na)  . del2(c)
    Ra += -2.0*D*k2 * (Na_xx+Na_yy) * (c_xx+c_yy);
    // D . del3(Na) . del3(c)
    Ra += D * ( (Na_xxx+Na_xyy)*(c_xxx+c_xyy) + (Na_xxy+Na_yyy)*(c_xxy+c_yyy) );
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

  PetscScalar c1[2];
  IGAPointFormGrad(p,U,&c1[0]);
  PetscScalar c_x   = c1[0],       c_y   = c1[1];

  const PetscReal *N0,(*N1)[2],(*N2)[2][2],(*N3)[2][2][2];
  IGAPointGetShapeFuns(p,0,(const PetscReal**)&N0);
  IGAPointGetShapeFuns(p,1,(const PetscReal**)&N1);
  IGAPointGetShapeFuns(p,2,(const PetscReal**)&N2);
  IGAPointGetShapeFuns(p,3,(const PetscReal**)&N3);

  PetscReal k4,k2 = user->k*user->k; k4 = k2*k2;
  PetscReal D = user->D, eps = user->Eps, g = user->g;

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
      PetscScalar t1 = -eps - 2.0*g*c + 3.0*c*c + D*k4;
      Kab += Na_x * t1 * Nb_x;
      Kab += Na_y * t1 * Nb_y;
      // grad(Na) . (deriv(t1)) grad(c)
      PetscScalar t2 = Nb*(-2.0*g + 6.0*c);
      Kab += Na_x * t2 * c_x;
      Kab += Na_y * t2 * c_y;
      // -2Dk^2 * del2(Na) * del2(Nb)
      Kab += -2.0*D*k2*(Na_xx+Na_yy) * (Nb_xx+Nb_yy);
      // D * del3(Na) * del3(Nb)
      Kab += user->D * ( (Na_xxx+Na_xyy)*(Nb_xxx+Nb_xyy) + (Na_xxy+Na_yyy)*(Nb_xxy+Nb_yyy) );
      /* ----- */
      K[a*nen+b] = Kab;
    }
  }
  return 0;
}

PetscReal FormRhoinit(PetscReal x ,PetscReal y, AppCtx *user){
  PetscReal rhoinit = user->cbar,xl,yl;
  PetscInt i;
  for(i=0;i<3;i++){
    PetscReal d1 = sqrt(SQ(x-user->C1[0])+SQ(y-user->C1[1]));
    PetscReal d2 = sqrt(SQ(x-user->C2[0])+SQ(y-user->C2[1]));
    PetscReal d3 = sqrt(SQ(x-user->C3[0])+SQ(y-user->C3[1]));
    if (d1 < user->dist){
      xl = cos(user->ang[i])*x+sin(user->ang[i])*y;
      yl = -sin(user->ang[i])*x+cos(user->ang[i])*y;
      return rhoinit + user->coefC*(cos(user->coefq/sqrt(3.)*yl)*cos(user->coefq*xl)-0.5*cos(2.*user->coefq/sqrt(3.)*yl));
    }
    if (d2 < user->dist){
      xl = cos(user->ang[i])*x+sin(user->ang[i])*y;
      yl = -sin(user->ang[i])*x+cos(user->ang[i])*y;
      return rhoinit + user->coefC*(cos(user->coefq/sqrt(3.)*yl)*cos(user->coefq*xl)-0.5*cos(2.*user->coefq/sqrt(3.)*yl));
    }
    if (d3 < user->dist){
      xl = cos(user->ang[i])*x+sin(user->ang[i])*y;
      yl = -sin(user->ang[i])*x+cos(user->ang[i])*y;
      return rhoinit + user->coefC*(cos(user->coefq/sqrt(3.)*yl)*cos(user->coefq*xl)-0.5*cos(2.*user->coefq/sqrt(3.)*yl));
    }
  }
  return rhoinit;
}

typedef struct {
  PetscScalar rho;
} Field;

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
  } else { /* initial condition is like Hectors initial test */
    DM da;
    ierr = IGACreateNodeDM(iga,1,&da);CHKERRQ(ierr);
    Field **u;
    ierr = DMDAVecGetArray(da,C,&u);CHKERRQ(ierr);
    DMDALocalInfo info;
    ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
    PetscInt i,j;
    for(i=info.xs;i<info.xs+info.xm;i++){
      for(j=info.ys;j<info.ys+info.ym;j++){
	PetscReal x = (PetscReal)i/((PetscReal)info.mx-1)*user->L0;
	PetscReal y = (PetscReal)j/((PetscReal)info.my-1)*user->L0;
	u[j][i].rho = FormRhoinit(x,y,user); 
      }
    }
    ierr = DMDAVecRestoreArray(da,C,&u);CHKERRQ(ierr); 
    ierr = DMDestroy(&da);CHKERRQ(ierr); 
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
  ierr = WriteSolution(U,"pfc%d.dat",step);CHKERRQ(ierr);
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

  user.C1[0]=200.0;user.C1[1]=200.0;
  user.C2[0]=600.0;user.C2[1]=200.0;
  user.C3[0]=400.0;user.C3[1]=700.0;
  user.ang[0]=-0.25*PETSC_PI; user.ang[0]=0.0; user.ang[2]=0.25*PETSC_PI;
  user.dist=20.0;
  user.coefC=0.466;
  user.coefq=0.66; 

  /* Set discretization options */
  PetscInt N=1024, p=3, C=PETSC_DECIDE;
  PetscBool output = PETSC_FALSE; 
  PetscBool monitor = PETSC_FALSE; 
  char initial[PETSC_MAX_PATH_LEN] = {0};
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","PhaseFieldCrystal2D Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-N","number of elements (along one dimension)",__FILE__,N,&N,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-p","polynomial order",__FILE__,p,&p,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-C","global continuity order",__FILE__,C,&C,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-pfc_initial","Load initial solution from file",__FILE__,initial,initial,sizeof(initial),PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pfc_output","Enable output files",__FILE__,output,&output,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pfc_monitor","Compute and show statistics of solution",__FILE__,monitor,&monitor,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-pfc_cbar","Initial atomistic density field",__FILE__,user.cbar,&user.cbar,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-pfc_g","Physical parameter",__FILE__,user.g,&user.g,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-pfc_k","Positive number",__FILE__,user.k,&user.k,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-pfc_Eps","Physical parameter",__FILE__,user.Eps,&user.Eps,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-pfc_D","Positive number",__FILE__,user.D,&user.D,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (C == PETSC_DECIDE) C = p-1;
  
  if (p < 3 || C < 2) /* Problem requires a p>=2 C1 basis */
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,
	    "Problem requires minimum of p = 3 and C = 2");
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
  ierr = TSSetDuration(ts,10000000,10000.0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,1.);CHKERRQ(ierr);

  ierr = TSSetType(ts,TSALPHA);CHKERRQ(ierr);
  ierr = TSAlphaSetRadius(ts,1.0);CHKERRQ(ierr);

  if (monitor) {
    user.iga = iga;
    PetscPrintf(PETSC_COMM_WORLD,"#Time        dt           Free Energy\n");
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
