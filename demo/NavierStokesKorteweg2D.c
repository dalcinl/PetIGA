/* 
   This code solve the dimensionless form of the isothermal
   Navier-Stokes-Korteweg equations as presented in:
   
   Gomez, Hughes, Nogueira, Calo
   Isogeometric analysis of the isothermal Navier-Stokes-Korteweg equations
   CMAME, 2010
   
   Equation/section numbers reflect this publication.
*/
#include "petiga.h"
#define SQ(x) ((x)*(x))

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

  // interpolate solution vector at current pnt
  PetscScalar sol_t[3],sol[3];
  PetscScalar grad_sol[3][2];
  PetscScalar hess_sol[3][2][2];
  IGAPointFormValue(pnt,V,&sol_t[0]);
  IGAPointFormValue(pnt,U,&sol[0]);
  IGAPointFormGrad (pnt,U,&grad_sol[0][0]);
  IGAPointFormHess (pnt,U,&hess_sol[0][0][0]);

  // now load these into more readable variables
  PetscScalar rho, rho_t, grad_rho[2], div_rho;
  rho          = sol[0]; 
  rho_t        = sol_t[0]; 
  grad_rho[0]  = grad_sol[0][0]; 
  grad_rho[1]  = grad_sol[0][1]; 
  div_rho      = hess_sol[0][0][0]+hess_sol[0][1][1];

  PetscScalar u[2], u_t[2], grad_u[2][2];
  u[0]         = sol[1]; 
  u[1]         = sol[2]; 
  u_t[0]       = sol_t[1]; 
  u_t[1]       = sol_t[2]; 
  grad_u[0][0] = grad_sol[1][0]; 
  grad_u[0][1] = grad_sol[1][1]; 
  grad_u[1][0] = grad_sol[2][0]; 
  grad_u[1][1] = grad_sol[2][1]; 

  PetscReal p = 8.0/27.0*theta*rho/(1.0-rho)-rho*rho; 

  const PetscReal *N0,(*N1)[2],(*N2)[2][2];
  IGAPointGetShapeFuns(pnt,0,(const PetscReal**)&N0);
  IGAPointGetShapeFuns(pnt,1,(const PetscReal**)&N1);
  IGAPointGetShapeFuns(pnt,2,(const PetscReal**)&N2);
  
  PetscScalar (*R)[3] = (PetscScalar (*)[3])Re;
  PetscInt a,i,nen=pnt->nen;
  for(a=0; a<nen; a++) {
    
    PetscReal R_c;    
    R_c = N0[a]*rho_t-rho*(N1[a][0]*u[0]+N1[a][1]*u[1]);
   
    PetscReal R_m[2];
    for(i=0;i<2;i++){
      R_m[i]  =  N0[a]*(rho_t*u[i]+rho*u_t[i]);
      R_m[i] += -rho*u[i]*(N1[a][0]*u[0]+N1[a][1]*u[1]);
      R_m[i] += -N1[a][i]*p;
      R_m[i] +=  rRe*(N1[a][0]*grad_u[i][0]+N1[a][1]*grad_u[i][1]);
      R_m[i] +=  rRe*(N1[a][0]*grad_u[0][i]+N1[a][1]*grad_u[1][i]);
      R_m[i] += -(2./3.)*rRe*N1[a][i]*(grad_u[0][0]+grad_u[1][1]);
      R_m[i] +=  Ca2*N1[a][i]*rho*div_rho;
      R_m[i] +=  0.5*Ca2*N1[a][i]*(grad_rho[0]*grad_rho[0]+grad_rho[1]*grad_rho[1]);
      R_m[i] += -Ca2*grad_rho[i]*(N1[a][0]*grad_rho[0]+N1[a][1]*grad_rho[1]);
    }

    R[a][0] = R_c;
    R[a][1] = R_m[0];
    R[a][2] = R_m[1];
  }
  return 0;
}

#undef  __FUNCT__
#define __FUNCT__ "Jacobian"
PetscErrorCode Jacobian(IGAPoint pnt,PetscReal dt,
                        PetscReal shift,const PetscScalar *V,
                        PetscReal t,const PetscScalar *U,
                        PetscScalar *Je,void *ctx)
{
  AppCtx *user = (AppCtx*) ctx;
  PetscReal Ca2 = user->Ca*user->Ca;
  PetscReal rRe = 1.0/user->Re;
  PetscReal theta = user->theta;

  // interpolate solution vector at current pnt
  PetscScalar sol_t[3],sol[3];
  PetscScalar grad_sol[3][2];
  PetscScalar hess_sol[3][2][2];
  IGAPointFormValue(pnt,V,&sol_t[0]);
  IGAPointFormValue(pnt,U,&sol[0]);
  IGAPointFormGrad (pnt,U,&grad_sol[0][0]);
  IGAPointFormHess (pnt,U,&hess_sol[0][0][0]);

  // density variables
  PetscScalar rho, rho_t, grad_rho[2];
  rho         = sol[0]; 
  rho_t       = sol_t[0]; 
  grad_rho[0] = grad_sol[0][0]; 
  grad_rho[1] = grad_sol[0][1]; 
  PetscScalar div_rho = hess_sol[0][0][0] + hess_sol[0][1][1];

  // velocity variables
  PetscScalar u[2], u_t[2];
  u[0]   = sol[1];   u[1]   = sol[2]; 
  u_t[0] = sol_t[1]; u_t[1] = sol_t[2]; 

  // pressure
  PetscReal dp = 8./27.*theta/(1.-rho)*(rho/(1.-rho)+1.)-2.*rho;

  const PetscReal *N0,(*N1)[2],(*N2)[2][2];
  IGAPointGetShapeFuns(pnt,0,(const PetscReal**)&N0);
  IGAPointGetShapeFuns(pnt,1,(const PetscReal**)&N1);
  IGAPointGetShapeFuns(pnt,2,(const PetscReal**)&N2);
  
  PetscInt a,b,i,j,nen=pnt->nen;
  PetscScalar (*J)[3][nen][3] = (PetscScalar (*)[3][nen][3])Je;
  for(a=0; a<nen; a++) {
    for(b=0; b<nen; b++) {

      // continuity jacobian
      J[a][0][b][0] += shift*N0[a]*N0[b];
      for(i=0; i<2; i++) {
	J[a][0][b][0] += -N1[a][i]*N0[b]*u[i];
	J[a][0][b][1+i] += -N1[a][i]*rho*N0[b];
      }
      
      // momentum jacobian
      PetscScalar divN = N2[b][0][0]+N2[b][1][1];
      for(i=0; i<2; i++) {
	J[a][1+i][b][0] +=  shift*N0[a]*N0[b]*u[i];
	J[a][1+i][b][0] +=  N0[a]*N0[b]*u_t[i];
	J[a][1+i][b][0] += -N0[b]*u[i]*(N1[a][0]*u[0]+N1[a][1]*u[1]);
	J[a][1+i][b][0] += -N1[a][i]*dp*N0[b];
	J[a][1+i][b][0] +=  Ca2*N1[a][i]*(N0[b]*div_rho+rho*divN);
	J[a][1+i][b][0] +=  Ca2*N1[a][i]*(grad_rho[0]*N1[b][0]+grad_rho[1]*N1[b][1]);
	J[a][1+i][b][0] += -Ca2*N1[b][i]*(grad_rho[0]*N1[a][0]+grad_rho[1]*N1[a][1]);
	J[a][1+i][b][0] += -Ca2*grad_rho[i]*(N1[a][0]*N1[b][0]+N1[a][1]*N1[b][1]);
	for(j=0; j<2; j++) {
	  if(i==j) {
	    J[a][1+i][b][1+j] +=  shift*N0[a]*rho*N0[b];
	    J[a][1+i][b][1+j] +=  N0[a]*rho_t*N0[b];
	    J[a][1+i][b][1+j] += -rho*N0[b]*(N1[a][0]*u[0]+N1[a][1]*u[1]);
	    J[a][1+i][b][1+j] +=  rRe*(N1[a][0]*N1[b][0]+N1[a][1]*N1[b][1]);
	  }
	  J[a][1+i][b][1+j] += -N1[a][j]*rho*u[i]*N0[b];
	  J[a][1+i][b][1+j] +=  rRe*N1[a][j]*N1[b][i];
	  J[a][1+i][b][1+j] += -2./3.*rRe*N1[a][i]*N1[b][j];
	}
      }
    }
  }
  return 0;
}

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
				   tanh(0.5*(d3-user->R3)/user->Ca));
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
  sprintf(filename,"./nsk2d%d.dat",it_number);
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

  PetscScalar scalar = 0.;
  ierr = IGAFormScalar(user->iga,U,1,&scalar,Energy,mctx);CHKERRQ(ierr);
  PetscReal energy = PetscRealPart(scalar);

  PetscReal dt;
  TSGetTimeStep(ts,&dt);

  if(step > 0 && energy > user->energy) {
    PetscPrintf(PETSC_COMM_WORLD,"%.6e %.6e %.16e  WARNING: Free energy increased!\n",t,dt,energy);
  }else{
    PetscPrintf(PETSC_COMM_WORLD,"%.6e %.6e %.16e\n",t,dt,energy);
  }
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
  user.C1[0] = 0.75; user.C1[1] = 0.50; 
  user.C2[0] = 0.25; user.C2[1] = 0.50; 
  user.C3[0] = 0.40; user.C3[1] = 0.75; 
  user.R1 = 0.10; user.R2 = 0.15; user.R3 = 0.08;

  user.alpha = 2.0; // (Eq. 41)
  user.theta = 0.85; // temperature parameter (just before section 5.1)

  // Set discretization options
  PetscInt  N=256,p=2,C=1;
  PetscBool output=PETSC_FALSE,monitor=PETSC_TRUE;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "NSK Options", "IGA");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-N", "number of elements along one dimension", __FILE__, N, &N, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-p", "polynomial order", __FILE__, p, &p, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-C", "global continuity order", __FILE__, C, &C, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-nsk_output","Enable output files",__FILE__,output,&output,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-nsk_monitor","Monitor the free energy of the solution",__FILE__,monitor,&monitor,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  // Compute simulation parameters
  user.h  = user.L0/N; // characteristic length scale of mesh (Eq. 43, simplified for uniform elements)
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
  IGAAxis axis;
  ierr = IGAGetAxis(iga,1,&axis);CHKERRQ(ierr);
  ierr = IGAAxisCopy(axis0,axis);CHKERRQ(ierr);
  
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);
  ierr = IGAWrite(iga,"iga.dat");CHKERRQ(ierr);
  user.iga = iga;

  ierr = IGASetUserIFunction(iga,Residual,&user);CHKERRQ(ierr);
  ierr = IGASetUserIJacobian(iga,Jacobian,&user);CHKERRQ(ierr);

  TS ts;
  ierr = IGACreateTS(iga,&ts);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,1000000,1000.0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,1.0e-2);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSALPHA);CHKERRQ(ierr);
  ierr = TSAlphaSetRadius(ts,0.5);CHKERRQ(ierr);
  if (monitor) {ierr = TSMonitorSet(ts,NSKMonitor,&user,NULL);CHKERRQ(ierr);}
  if (output)  {ierr = TSMonitorSet(ts,OutputMonitor,&user,NULL);CHKERRQ(ierr);}
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  PetscReal t=0; Vec U;
  ierr = IGACreateVec(iga,&U);CHKERRQ(ierr);
  ierr = FormInitialCondition(iga,t,U,&user);CHKERRQ(ierr);
#if PETSC_VERSION_LE(3,3,0)
  ierr = TSSolve(ts,U,NULL);CHKERRQ(ierr);
#else
  ierr = TSSolve(ts,U);CHKERRQ(ierr);
#endif

  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}



