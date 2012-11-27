/* This code solve the dimensionless form of the isothermal
   Navier-Stokes-Korteweg equations as presented in:
   
   Gomez, Hughes, Nogueira, Calo
   Isogeometric analysis of the isothermal Navier-Stokes-Korteweg equations
   CMAME, 2010
   
   Equation/section numbers reflect this publication.
*/
#include "petiga.h"

typedef struct {
  IGA       iga;
  PetscReal energy;
  // problem parameters
  PetscReal L0,h;
  PetscReal Ca,alpha,theta,Re;
  // bubble centers
  PetscReal C1[2],R1;
  PetscReal C2[2],R2;
  PetscReal C3[2],R3;
} AppCtx;

#undef  __FUNCT__
#define __FUNCT__ "Residual"
PetscErrorCode Residual(IGAPoint pnt,PetscReal dt,
                        PetscReal shift,const PetscScalar *V,
                        PetscReal t,const PetscScalar *U,
                        PetscScalar *Re,void *ctx)
{
  AppCtx *user = (AppCtx*) ctx;
  PetscReal Ca2 = user->Ca*user->Ca;
  PetscReal rRe = 1.0/user->Re;
  PetscReal theta = user->theta;

  PetscScalar u_t[3],u[3];
  PetscScalar grad_u[3][2];
  PetscScalar hess_u[3][2][2];
  IGAPointFormValue(pnt,V,&u_t[0]);
  IGAPointFormValue(pnt,U,&u[0]);
  IGAPointFormGrad (pnt,U,&grad_u[0][0]);
  IGAPointFormHess (pnt,U,&hess_u[0][0][0]);
  PetscReal rho_t  = u_t[0], rho = u[0];
  PetscReal rho_x  = grad_u[0][0],    rho_y  = grad_u[0][1];
  PetscReal rho_xx = hess_u[0][0][0], rho_xy = hess_u[0][0][1];
  PetscReal rho_yx = hess_u[0][1][0], rho_yy = hess_u[0][1][1];
  PetscReal ux_t  = u_t[1], ux = u[1];
  PetscReal uy_t  = u_t[2], uy = u[2];
  PetscReal ux_x = grad_u[1][0], ux_y = grad_u[1][1];
  PetscReal uy_x = grad_u[2][0], uy_y = grad_u[2][1];

  // compute pressure (Eq. 34.3)
  PetscReal p = 8.0/27.0*theta*rho/(1.0-rho)-rho*rho; 
  // compute viscous stress tensor (Eq. 34.4)
  PetscReal tau_xx = 2.0*ux_x - 2.0/3.0*(ux_x+uy_y);
  PetscReal tau_xy = ux_y + uy_x ;
  PetscReal tau_yy = 2.0*uy_y - 2.0/3.0*(ux_x+uy_y);
  PetscReal tau_yx = tau_xy;

  const PetscReal *N0,(*N1)[2],(*N2)[2][2];
  IGAPointGetShapeFuns(pnt,0,(const PetscReal**)&N0);
  IGAPointGetShapeFuns(pnt,1,(const PetscReal**)&N1);
  IGAPointGetShapeFuns(pnt,2,(const PetscReal**)&N2);
  
  PetscScalar (*R)[3] = (PetscScalar (*)[3])Re;
  PetscInt a,nen=pnt->nen;
  for(a=0; a<nen; a++) {
    PetscReal Na    = N0[a];
    PetscReal Na_x  = N1[a][0];
    PetscReal Na_y  = N1[a][1];
    PetscReal Na_xx = N2[a][0][0];
    PetscReal Na_yy = N2[a][1][1];
    PetscReal Na_xy = N2[a][1][2];
    
    PetscReal R_rho, R_ux, R_uy;
    // (Eq. 19, modified to be dimensionless)
    R_rho  = Na*rho_t; 
    R_rho += -rho*(Na_x*ux + Na_y*uy);
    
    R_ux  = Na*ux*rho_t; 
    R_ux += Na*rho*ux_t; 
    R_ux += -rho*(Na_x*ux*ux + Na_y*ux*uy);
    R_ux += -Na_x*p;
    R_ux += rRe*(Na_x*tau_xx + Na_y*tau_xy);
    R_ux += -Ca2*rho*(Na_xx*rho_x + Na_xy*rho_y); 
    R_ux += -Ca2*Na_x*(rho_x*rho_x + rho_y*rho_y);
    R_ux += -Ca2*Na*(rho_xx*rho_x + rho_xy*rho_y);
    R_ux += -Ca2*rho_x*(Na_x*rho_x + Na_y*rho_y);
            
    R_uy  = Na*uy*rho_t; 
    R_uy += Na*rho*uy_t; 
    R_uy += -rho*(Na_x*uy*ux + Na_y*uy*uy);
    R_uy += -Na_y*p;
    R_uy += rRe*(Na_x*tau_yx + Na_y*tau_yy);
    R_uy += -Ca2*rho*(Na_xy*rho_x + Na_yy*rho_y);
    R_uy += -Ca2*Na_y*(rho_x*rho_x + rho_y*rho_y);
    R_uy += -Ca2*Na*(rho_yx*rho_x + rho_yy*rho_y);
    R_uy += -Ca2*rho_y*(Na_x*rho_x + Na_y*rho_y);

    R[a][0] = R_rho;
    R[a][1] = R_ux;
    R[a][2] = R_uy;
  }
  return 0;
}

#define SQ(x) ((x)*(x))

typedef struct {
  PetscScalar rho,ux,uy;
} Field;

#undef __FUNCT__
#define __FUNCT__ "FormInitialCondition"
PetscErrorCode FormInitialCondition(IGA iga,PetscReal t,Vec U,AppCtx *user)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  DM da;
  ierr = IGACreateNodeDM(iga,3,&da);CHKERRQ(ierr);
  Field **u;
  ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);
  DMDALocalInfo info;
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);

  PetscInt i,j;
  for(i=info.xs;i<info.xs+info.xm;i++){
    for(j=info.ys;j<info.ys+info.ym;j++){
      PetscReal x = (PetscReal)i / ( (PetscReal)(info.mx-1) );
      PetscReal y = (PetscReal)j / ( (PetscReal)(info.my-1) );
      PetscReal d1 = sqrt(SQ(x-user->C1[0])+SQ(y-user->C1[1]));
      PetscReal d2 = sqrt(SQ(x-user->C2[0])+SQ(y-user->C2[1]));
      PetscReal d3 = sqrt(SQ(x-user->C3[0])+SQ(y-user->C3[1]));
      
      u[j][i].rho = -0.15 + 0.25*( tanh(0.5*(d1-user->R1)/user->Ca) +
                                   tanh(0.5*(d2-user->R2)/user->Ca) +
                                   tanh(0.5*(d3-user->R3)/user->Ca) );
      u[j][i].ux = 0.0;
      u[j][i].uy = 0.0;
    }
  }
  ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr); 
  ierr = DMDestroy(&da);;CHKERRQ(ierr); 
  PetscFunctionReturn(0); 
}

#undef __FUNCT__
#define __FUNCT__ "OutputMonitor"
PetscErrorCode OutputMonitor(TS ts,PetscInt it_number,PetscReal c_time,Vec U,void *mctx)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  AppCtx *user = (AppCtx *)mctx;
  char           filename[256];
  sprintf(filename,"./nsk%d.dat",it_number);
  ierr = IGAWriteVec(user->iga,U,filename);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FreeEnergy"
PetscErrorCode FreeEnergy(PetscScalar rho,PetscScalar rho_x,PetscScalar rho_y,PetscScalar ux,PetscScalar uy,PetscScalar *E_tmp,AppCtx *user)
{
  PetscFunctionBegin;
  *E_tmp = 8.0/27.0*user->theta*rho*log(rho/(1.0-rho))-rho*rho;
  *E_tmp += 0.5*user->Ca*user->Ca*(rho_x*rho_x+rho_y*rho_y);
  *E_tmp += 0.5*(ux*ux+uy*uy);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "Energy"
PetscErrorCode Energy(IGAPoint pnt,const PetscScalar *U,PetscInt n,PetscScalar *S,void *ctx)
{
  PetscFunctionBegin;
  AppCtx *user = (AppCtx *)ctx;
  PetscScalar u[3];
  PetscScalar grad_u[3][2];
  IGAPointFormValue(pnt,U,&u[0]);
  IGAPointFormGrad (pnt,U,&grad_u[0][0]);
  PetscReal rho    = u[0], ux = u[1], uy = u[2];
  PetscReal rho_x  = grad_u[0][0];
  PetscReal rho_y  = grad_u[0][1];
  FreeEnergy(rho,rho_x,rho_y,ux,uy,S,user);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NSKMonitor"
PetscErrorCode NSKMonitor(TS ts,PetscInt step,PetscReal t,Vec U,void *mctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  AppCtx *user = (AppCtx *)mctx;

  PetscScalar energy = 0.;
  ierr = IGAFormScalar(user->iga,U,1,&energy,Energy,mctx);CHKERRQ(ierr);

  PetscReal dt;
  TSGetTimeStep(ts,&dt);
  PetscPrintf(PETSC_COMM_WORLD,"%.6e %.6e %.16e\n",t,dt,energy);

  if(step > 0 && energy > user->energy) PetscPrintf(PETSC_COMM_WORLD,"WARNING: Free energy increased!\n");
  user->energy = energy; 

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[]) {

  // Petsc Initialization rite of passage 
  PetscErrorCode  ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  // Define simulation specific parameters
  AppCtx user;
  user.L0 = 1.0; // length scale
  user.C1[0] = 0.75; user.C1[1] = 0.50; // bubble centers
  user.C2[0] = 0.25; user.C2[1] = 0.50;
  user.C3[0] = 0.40; user.C3[1] = 0.75;
  user.R1 = 0.10;  user.R2 = 0.15;  user.R3 = 0.08; // bubble radii

  user.alpha = 2.0; // (Eq. 41)
  user.theta = 0.85; // temperature parameter (just before section 5.1)

  // Set discretization options
  PetscInt  N=64,p=2,C=1;
  PetscBool output=PETSC_FALSE,monitor=PETSC_FALSE;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "NSK Options", "IGA");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-N", "number of elements along one dimension", __FILE__, N, &N, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-p", "polynomial order", __FILE__, p, &p, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-C", "global continuity order", __FILE__, C, &C, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-nsk_output","Enable output files",__FILE__,output,&output,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-nsk_monitor","Monitor the free energy of the solution",__FILE__,monitor,&monitor,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  // Compute simulation parameters
  user.h = user.L0/N; // characteristic length scale of mesh (Eq. 43, simplified for uniform elements)
  user.Ca = user.h/user.L0; // capillarity number (Eq. 38)
  user.Re = user.alpha/user.Ca; // Reynolds number (Eq. 39)

  // Problem requires a C1 basis
  if (p < 2 || C < 1) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Problem requires minimum of p = 2 and C = 1");
  }
  // Test C < p
  if (p <= C) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Discretization inconsistent: polynomial order must be greater than degree of continuity");
  }

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDim(iga,2);CHKERRQ(ierr);
  ierr = IGASetDof(iga,3);CHKERRQ(ierr);
  ierr = IGASetFieldName(iga,0,"density"); CHKERRQ(ierr);
  ierr = IGASetFieldName(iga,1,"velocity-u"); CHKERRQ(ierr);
  ierr = IGASetFieldName(iga,2,"velocity-v"); CHKERRQ(ierr);

  IGAAxis axis0;
  ierr = IGAGetAxis(iga,0,&axis0);CHKERRQ(ierr);
  ierr = IGAAxisSetPeriodic(axis0,PETSC_TRUE);CHKERRQ(ierr);
  ierr = IGAAxisSetDegree(axis0,p);CHKERRQ(ierr);
  ierr = IGAAxisInitUniform(axis0,N,0.0,1.0,C);CHKERRQ(ierr);
  IGAAxis axis1;
  ierr = IGAGetAxis(iga,1,&axis1);CHKERRQ(ierr);
  ierr = IGAAxisCopy(axis0,axis1);CHKERRQ(ierr);
  
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);
  user.iga = iga;

  ierr = IGASetUserIFunction(iga,Residual,&user);CHKERRQ(ierr);

  TS ts;
  ierr = IGACreateTS(iga,&ts);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,1000000,1000.0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,1.0e-2);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSALPHA);CHKERRQ(ierr);
  ierr = TSAlphaSetRadius(ts,0.5);CHKERRQ(ierr);

  if (monitor) {ierr = TSMonitorSet(ts,NSKMonitor,&user,PETSC_NULL);CHKERRQ(ierr);}
  if (output)  {ierr = TSMonitorSet(ts,OutputMonitor,&user,PETSC_NULL);CHKERRQ(ierr);}
  ierr = PetscOptionsSetValue("-snes_mf","true");CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  PetscReal t=0; Vec U;
  ierr = IGACreateVec(iga,&U);CHKERRQ(ierr);
  ierr = FormInitialCondition(iga,t,U,&user);CHKERRQ(ierr);
#if PETSC_VERSION_(3,3,0) || PETSC_VERSION_(3,2,0)
    ierr = TSSolve(ts,U,PETSC_NULL);CHKERRQ(ierr);
#else
    ierr = TSSolve(ts,U);CHKERRQ(ierr);
#endif

  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}



