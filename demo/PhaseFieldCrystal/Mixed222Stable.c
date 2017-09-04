#include "petiga.h"

typedef struct {
  PetscReal eps;
  PetscReal phi_bar;
  PetscReal alpha;
  PetscReal energy;
} AppCtx;

PetscErrorCode FreeEnergyMass(IGAPoint p,const PetscScalar *U,PetscInt n,PetscScalar *S,void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;
  PetscFunctionBegin;

  PetscInt i;
  PetscInt dim = p->dim;
  PetscScalar c[3],c1[3][dim],gc_dot_gc=0;
  IGAPointFormValue(p,U,&c[0]);
  IGAPointFormGrad (p,U,&c1[0][0]);

  for (i=0; i<dim; i++)
    gc_dot_gc += c1[0][i]* c1[0][i];

  // Free energy
  S[0] = 0.25*c[0]*c[0]*c[0]*c[0] - 0.5*user->eps*c[0]*c[0] + 0.5*(c[0]*c[0] - 2.*gc_dot_gc + (c[2]-c[0])*(c[2]-c[0]));

  // Mass
  S[1] = c[0];

  PetscFunctionReturn(0);
}

PetscErrorCode EnergyMonitor(TS ts,PetscInt step,PetscReal t,Vec U,void *mctx)
{
  AppCtx *user = (AppCtx *)mctx;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  IGA iga;
  ierr = PetscObjectQuery((PetscObject)ts,"IGA",(PetscObject*)&iga);CHKERRQ(ierr);

  SNES     snes;
  PetscInt snes_it,ksp_it;
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&snes_it);CHKERRQ(ierr);
  ierr = SNESGetLinearSolveIterations(snes,&ksp_it);CHKERRQ(ierr);

  PetscScalar result[2];
  ierr = IGAComputeScalar(iga,U,2,result,FreeEnergyMass,mctx);CHKERRQ(ierr);
  PetscScalar energy = result[0];
  PetscScalar mass   = result[1];

  if (step == 0) {
    PetscPrintf(PETSC_COMM_WORLD,"#Time          SNES    KSP  Mass                    Free Energy\n");
  }
  if (step > 0 && energy > user->energy) {
    PetscPrintf(PETSC_COMM_WORLD,"%.6e %6d %6d  %.16e WARNING: free energy increased!\n",t,snes_it,ksp_it,energy);
  } else {
    PetscPrintf(PETSC_COMM_WORLD,"%.6e %6d %6d  %.16e  %.16e\n",t,snes_it,ksp_it,mass,energy);
  }
  user->energy = energy;

  PetscFunctionReturn(0);
}

PetscErrorCode Residual2nd2nd2nd(IGAPoint p,PetscReal dt,
                                 PetscReal shift,const PetscScalar *V,
                                 PetscReal t,const PetscScalar *U,
                                 PetscScalar *R,void *ctx);
PetscErrorCode Jacobian2nd2nd2nd(IGAPoint p,PetscReal dt,
                                 PetscReal shift,const PetscScalar *V,
                                 PetscReal t,const PetscScalar *U,
                                 PetscScalar *K,void *ctx);

#define SQ(A) ((A)*(A))

PetscReal TriangularLattice(const PetscReal x[],PetscReal phi_bar, PetscReal eps)
{
  PetscReal q = sqrt(3.0)/2.0;
  PetscReal xl = x[0], yl = x[1];

  PetscReal cx = 5*2*PETSC_PI/q;
  PetscReal cy = 6*sqrt(3.)*PETSC_PI/q;

  PetscReal d0 = 0.33*cx;
  PetscReal d = sqrt((xl-cx)*(xl-cx)+(yl-cy)*(yl-cy));

  PetscReal phi = phi_bar;

  if(d < d0) {
    PetscReal w = SQ(1.0-SQ(d/d0));
    PetscReal A = 0.8*(phi_bar+sqrt(15*eps-36*SQ(phi_bar))/3.0);
    PetscReal phi_S = (cos(yl*q/sqrt(3.))*cos(xl*q-PETSC_PI)-0.5*cos(yl*2*q/sqrt(3.)));
    phi += w*A*phi_S;
  }

  return phi;
}

PetscErrorCode ResidualL2Projection(IGAPoint p,const PetscScalar *U,PetscScalar *F,void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;
  PetscInt a,nen = p->nen;
  PetscInt   dim = p->dim;

  const PetscReal *N   = p->shape[0];
  PetscScalar (*Ra)[3] = (typeof(Ra)) F;

  PetscReal x[dim];
  IGAPointFormGeomMap(p,x);

  PetscScalar tmp[3];
  IGAPointFormValue(p,U,&tmp[0]);
  PetscScalar u  = tmp[0];
  PetscScalar u0 = TriangularLattice(x,user->phi_bar,user->eps);

  PetscScalar *V;
  IGAPointGetWorkVec(p,&V);
  Residual2nd2nd2nd(p,0.0,0.0,V,0.0,U,F,ctx);
  for (a=0; a<nen; a++)
    Ra[a][0] = N[a] * (u - u0);

  return 0;
}

PetscErrorCode JacobianL2Projection(IGAPoint p,const PetscScalar *U,PetscScalar *J,void *ctx)
{
  PetscInt a,b,c;
  PetscInt nen = p->nen;
  const PetscReal *N = p->shape[0];
  PetscScalar (*Jab)[3][nen][3] = (typeof(Jab)) J;

  PetscScalar *V;
  IGAPointGetWorkVec(p,&V);
  Jacobian2nd2nd2nd(p,0.0,0.0,V,0.0,U,J,ctx);
  for (a=0; a<nen; a++) {
    for (b=0; b<nen; b++) {
      Jab[a][0][b][0] = N[a] * N[b];
      for (c=1; c<3; c++)
        Jab[a][0][b][c] = 0.0;
    }
  }
  return 0;
}

PetscErrorCode FormInitialCondition(IGA iga,Vec U,AppCtx *user)
{
  SNES           snes;
  KSP            ksp;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = IGASetFormFunction(iga,ResidualL2Projection,user);CHKERRQ(ierr);
  ierr = IGASetFormJacobian(iga,JacobianL2Projection,user);CHKERRQ(ierr);
  ierr = IGACreateSNES(iga,&snes);CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(snes,"l2p_");CHKERRQ(ierr);
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp,1e-8,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPGMRES);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = SNESSolve(snes,NULL,U);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

void ChemicalPotential(AppCtx *user,PetscReal phi,PetscReal *psi,PetscReal *dpsi,PetscReal *d2psi)
{
  if (psi)   *psi   = phi*(phi*phi-user->eps);
  if (dpsi)  *dpsi  = 3.0*phi*phi-user->eps;
  if (d2psi) *d2psi = 6.0*phi;
}

PetscErrorCode Residual2nd2nd2nd(IGAPoint p,PetscReal dt,
                                 PetscReal shift,const PetscScalar *V,
                                 PetscReal t,const PetscScalar *U,
                                 PetscScalar *R,void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;
  PetscInt a,nen = p->nen;
  PetscInt i,dim = p->dim;

  PetscScalar u_t[3],u[3],u1[3][dim],u2[3][dim][dim];
  IGAPointFormValue(p,V,&u_t[0]);
  IGAPointFormValue(p,U,&u[0]);
  IGAPointFormGrad (p,U,&u1[0][0]);
  IGAPointFormHess (p,U,&u2[0][0][0]);
  PetscScalar phi_t = u_t[0];
  PetscScalar phi   = u[0],  *gphi = &u1[0][0]; // phi
  PetscScalar s     = u[1],  *gs   = &u1[1][0]; // sigma
  PetscScalar th    = u[2],  *gt   = &u1[2][0]; // theta

  const PetscReal  *N0       = (typeof(N0)) p->shape[0];
  const PetscReal (*N1)[dim] = (typeof(N1)) p->shape[1];

  PetscReal psi;
  ChemicalPotential(user,phi,&psi,NULL,NULL);

  PetscScalar lift = psi + th;
  PetscReal   gN_dot_gphi,gN_dot_gs,gN_dot_gt;
  PetscScalar (*Ra)[3] = (PetscScalar (*)[3])R;
  for (a=0; a<nen; a++) {
    gN_dot_gphi = 0;
    gN_dot_gs   = 0;
    gN_dot_gt   = 0;
    for (i=0; i<dim; i++) {
      gN_dot_gphi += N1[a][i]*gphi[i];
      gN_dot_gs   += N1[a][i]*gs[i];
      gN_dot_gt   += N1[a][i]*gt[i];
    }
    Ra[a][0] = N0[a]*phi_t + gN_dot_gs;
    Ra[a][1] = N0[a]*(s-lift) + gN_dot_gt;
    Ra[a][2] = N0[a]*(th-phi) + gN_dot_gphi;
  }
  return 0;
}

PetscErrorCode Jacobian2nd2nd2nd(IGAPoint p,PetscReal dt,
                                 PetscReal shift,const PetscScalar *V,
                                 PetscReal t,const PetscScalar *U,
                                 PetscScalar *K,void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;
  PetscInt a,b,nen = p->nen;
  PetscInt   i,dim = p->dim;

  PetscScalar u[3];
  IGAPointFormValue(p,U,&u[0]);
  PetscScalar phi = u[0];
  PetscReal dpsi;
  ChemicalPotential(user,phi,NULL,&dpsi,NULL);

  const PetscReal  *N0        = (typeof(N0)) p->shape[0];
  const PetscReal (*N1)[dim]  = (typeof(N1)) p->shape[1];
  PetscScalar (*J)[3][nen][3] = (typeof(J))  K;


  for (a=0; a<nen; a++) {
    for (b=0; b<nen; b++) {
      PetscReal gN_dot_gN = 0;
      for (i=0; i<dim; i++)
        gN_dot_gN += N1[a][i]*N1[b][i];

      J[a][0][b][0] +=  shift*N0[a]*N0[b];
      J[a][0][b][1] +=  gN_dot_gN;
      J[a][0][b][2] +=  0;
      J[a][1][b][0] += -N0[a]*N0[b]*dpsi;
      J[a][1][b][1] +=  N0[a]*N0[b];
      J[a][1][b][2] += -N0[a]*N0[b] + gN_dot_gN;
      J[a][2][b][0] += -N0[a]*N0[b] + gN_dot_gN;
      J[a][2][b][1] +=  0;
      J[a][2][b][2] +=  N0[a]*N0[b];
    }
  }
  return 0;
}

PetscErrorCode Residual2nd2nd2ndStable(IGAPoint p,PetscReal dt,
                                       PetscReal shift,const PetscScalar *V,
                                       PetscReal t,const PetscScalar *U1,
                                       PetscReal t0,const PetscScalar *U0,
                                       PetscScalar *R,void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;
  PetscReal alpha = user->alpha;
  PetscInt a,nen = p->nen;
  PetscInt i,dim = p->dim;

  // value at t0
  PetscScalar X[3],X1[3][dim];
  IGAPointFormValue(p,U0,&X[0]);
  IGAPointFormGrad (p,U0,&X1[0][0]);

  // value at t
  PetscScalar Y[3],Y1[3][dim];
  IGAPointFormValue(p,U1,&Y[0]);
  IGAPointFormGrad (p,U1,&Y1[0][0]);

  PetscScalar phi0 = X[0];
  PetscScalar phi1 = Y[0];

  PetscReal psi_c1,dpsi_c1,psi_e0,dpsi_e0;
  psi_e0  = user->eps*phi0;
  dpsi_e0 = user->eps;
  psi_c1  = phi1*phi1*phi1;
  dpsi_c1 = 3.0*phi1*phi1;

  PetscScalar phi_jump   =     phi1-phi0;
  PetscScalar phi_half   = 0.5*(phi1+phi0);
  PetscScalar *gphi0 = &X1[0][0], *gphi1 = &Y1[0][0];

  PetscScalar grad_phi_half[dim];
  PetscScalar grad_phi_jump[dim];
  for (i=0; i<dim; i++) {
    grad_phi_half[i]   = 0.5*(gphi1[i]+gphi0[i]);
    grad_phi_jump[i]   = (gphi1[i]-gphi0[i]);
  }

  PetscScalar   s = Y[1];
  PetscScalar *gs = &Y1[1][0];
  PetscScalar   l = Y[2];
  PetscScalar *gl = &Y1[2][0];

  const PetscReal  *N0       = (typeof(N0)) p->shape[0];
  const PetscReal (*N1)[dim] = (typeof(N1)) p->shape[1];
  PetscScalar     (*Ra)[3]   = (typeof(Ra)) R;

  for (a=0; a<nen; a++) {
    PetscScalar gN_dot_gs=0,gN_dot_gl=0,gN_dot_gphi_half=0;
    PetscScalar gN_dot_gphi_jump=0;
    for (i=0; i<dim; i++) {
      gN_dot_gs        += N1[a][i]*gs[i];
      gN_dot_gl        += N1[a][i]*gl[i];
      gN_dot_gphi_half += N1[a][i]*grad_phi_half[i];
      gN_dot_gphi_jump += N1[a][i]*grad_phi_jump[i];
    }
    Ra[a][0] = N0[a]*phi_jump/dt + gN_dot_gs;
    Ra[a][1] = N0[a]*(s-l-psi_c1+psi_e0+(dpsi_e0+dpsi_c1)*phi_jump*0.5) + gN_dot_gl - alpha*dt*gN_dot_gphi_jump;
    Ra[a][2] = N0[a]*(l-phi_half) + gN_dot_gphi_half;
  }
  return 0;
}

PetscErrorCode Jacobian2nd2nd2ndStable(IGAPoint p,PetscReal dt,
                                       PetscReal shift,const PetscScalar *V,
                                       PetscReal t,const PetscScalar *U1,
                                       PetscReal t0,const PetscScalar *U0,
                                       PetscScalar *K,void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;
  PetscReal alpha = user->alpha;
  PetscInt a,b,nen = p->nen;
  PetscInt   i,dim = p->dim;

  // value at t0
  PetscScalar X[3],X1[3][dim];
  IGAPointFormValue(p,U0,&X[0]);
  IGAPointFormGrad (p,U0,&X1[0][0]);

  // value at t
  PetscScalar Y[3],Y1[3][dim];
  IGAPointFormValue(p,U1,&Y[0]);
  IGAPointFormGrad (p,U1,&Y1[0][0]);

  PetscScalar phi0 = X[0];
  PetscScalar phi1 = Y[0];

  PetscReal dpsi_c1,d2psi_c1,dpsi_e0;
  dpsi_e0  = user->eps;
  dpsi_c1  = 3.0*phi1*phi1;
  d2psi_c1 = 6.0*phi1;

  PetscScalar phi_jump = phi1-phi0;

  const PetscReal  *N0        = (typeof(N0)) p->shape[0];
  const PetscReal (*N1)[dim]  = (typeof(N1)) p->shape[1];
  PetscScalar (*J)[3][nen][3] = (typeof(J))  K;

  for (a=0; a<nen; a++) {
    for (b=0; b<nen; b++) {
      PetscScalar gN_dot_gN = 0;
      for (i=0; i<dim; i++) {
        gN_dot_gN += N1[a][i]*N1[b][i];
      }
      J[a][0][b][0] += N0[a]*N0[b]/dt;
      J[a][0][b][1] += gN_dot_gN;
      J[a][0][b][2] += 0.0;
      J[a][1][b][0] += N0[a]*N0[b]*(-0.5*dpsi_c1+0.5*phi_jump*d2psi_c1+(0.5*dpsi_e0)) - alpha*dt*gN_dot_gN;
      J[a][1][b][1] += N0[a]*N0[b];
      J[a][1][b][2] += -N0[a]*N0[b] + gN_dot_gN;
      J[a][2][b][0] += -0.5*N0[a]*N0[b] + 0.5*gN_dot_gN;
      J[a][2][b][1] += 0.0;
      J[a][2][b][2] += N0[a]*N0[b];
    }
  }
  return 0;
}

PetscErrorCode OutputMonitor(TS ts,PetscInt step,PetscReal t,Vec U,void *mctx)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  IGA iga;
  ierr = PetscObjectQuery((PetscObject)ts,"IGA",(PetscObject*)&iga);CHKERRQ(ierr);
  char filename[256];
  sprintf(filename,"pfc%d.dat",(int)step);
  ierr = IGAWriteVec(iga,U,filename);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

int main(int argc, char *argv[]) {

  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  PetscInt  dim     = 2;
  PetscInt  N       = 64;
  PetscInt  p       = 1;
  PetscInt  k       = 0;

  PetscReal eps     = 0.325;
  PetscReal phi_bar = 0.5*sqrt(eps);
  PetscBool stable  = PETSC_TRUE;
  PetscReal alpha   = 0.25;

  PetscBool monitor = PETSC_TRUE;
  PetscBool output  = PETSC_FALSE;
  char final[PETSC_MAX_PATH_LEN] = "pfc_final.dat";

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","PhaseFieldCrystal Options","IGA");CHKERRQ(ierr);

  ierr = PetscOptionsInt("-dim","number of spatial dimensions",__FILE__,dim,&dim,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-N","number of elements in x direction",__FILE__,N,&N,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-p","polynomial degree",  __FILE__,p,&p,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-k","basis continuity",  __FILE__,k,&k,NULL);CHKERRQ(ierr);

  ierr = PetscOptionsReal("-epsilon","Degree of undercooling", __FILE__,eps,&eps,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-phi_bar","Liquid atomistic density ", __FILE__,phi_bar,&phi_bar,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-stable","Use energy-stable formulation",__FILE__,stable,&stable,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-alpha","Stabilization parameter", __FILE__,alpha,&alpha,NULL);CHKERRQ(ierr);

  ierr = PetscOptionsBool("-monitor","Monitor free energy and mass evolution",__FILE__,monitor,&monitor,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-output","Enable solution output files",__FILE__,output,&output,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-final","Filename for final solution",__FILE__,final,final,sizeof(final),NULL);CHKERRQ(ierr);

  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  if (p < 1) /* Problem requires a p>=1 basis */
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,
            "Problem requires minimum of p = 1");
  if (k < 0) /* Problem requires a C0 basis */
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,
            "Problem requires minimum of k = 0");
  if (p <= k)  /* Check k < p */
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,
            "Inconsistent discretization: polynomial order must be greater than degree of continuity");

  AppCtx user;
  user.eps     = eps;
  user.phi_bar = phi_bar;
  user.alpha   = alpha;

  // Setup domain size and problem parameters
  PetscReal L[3];
  PetscInt  E[3];
  PetscScalar q  = sqrt(3.)*0.5;
  PetscScalar Tx = 2*PETSC_PI/q;
  PetscScalar Ty = sqrt(3.)*PETSC_PI/q;
  L[0] = 10*Tx; E[0] = N;
  L[1] = 12*Ty; E[1] = (PetscInt)round(E[0]*L[1]/L[0]);

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDim(iga,dim);CHKERRQ(ierr);
  ierr = IGASetDof(iga,3);CHKERRQ(ierr);
  ierr = IGASetFieldName(iga,0,"phi");CHKERRQ(ierr);
  ierr = IGASetFieldName(iga,1,"sigma");CHKERRQ(ierr);
  ierr = IGASetFieldName(iga,2,"theta");CHKERRQ(ierr);
  PetscInt i;
  for (i=0; i<dim; i++) {
    IGAAxis  axis;
    ierr = IGAGetAxis(iga,i,&axis);CHKERRQ(ierr);
    ierr = IGAAxisSetPeriodic(axis,PETSC_TRUE);CHKERRQ(ierr);
    ierr = IGAAxisSetDegree(axis,p);CHKERRQ(ierr);
    ierr = IGAAxisInitUniform(axis,E[i],0.0,L[i],k);CHKERRQ(ierr);
  }
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);
  ierr = IGAWrite(iga,"pfc_iga.dat");CHKERRQ(ierr);

  TS ts;
  ierr = IGACreateTS(iga,&ts);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,150.0);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,0.5);CHKERRQ(ierr);
  if (stable) {
    ierr = TSSetType(ts,TSBEULER);CHKERRQ(ierr);
    ierr = IGASetFormIEFunction(iga,Residual2nd2nd2ndStable,&user);CHKERRQ(ierr);
    ierr = IGASetFormIEJacobian(iga,Jacobian2nd2nd2ndStable,&user);CHKERRQ(ierr);
  } else {
    ierr = TSSetType(ts,TSTHETA);CHKERRQ(ierr);
    ierr = IGASetFormIFunction(iga,Residual2nd2nd2nd,&user);CHKERRQ(ierr);
    ierr = IGASetFormIJacobian(iga,Jacobian2nd2nd2nd,&user);CHKERRQ(ierr);
  }

  if (monitor) {ierr = TSMonitorSet(ts,EnergyMonitor,&user,NULL);CHKERRQ(ierr);}
  if (output)  {ierr = TSMonitorSet(ts,OutputMonitor,&user,NULL);CHKERRQ(ierr);}
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  Vec U;
  ierr = IGACreateVec(iga,&U);CHKERRQ(ierr);
  ierr = FormInitialCondition(iga,U,&user);CHKERRQ(ierr);

  ierr = TSSolve(ts,U);CHKERRQ(ierr);

  ierr = IGAWriteVec(iga,U,final);CHKERRQ(ierr);

  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
