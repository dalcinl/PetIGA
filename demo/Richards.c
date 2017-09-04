#include "petiga.h"

#define SQ(A) ((A)*(A))

typedef struct {
  PetscReal m,lambda,kappa,NGr;
  PetscReal S0,Sin,d0,penalty;
  PetscBool phase_field;
} AppCtx;

PetscErrorCode L2Projection(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  if (p->atboundary) return 0;

  AppCtx *user = (AppCtx *)ctx;

  PetscInt nen = p->nen;
  PetscInt dim = p->dim;

  PetscReal x[dim];
  IGAPointFormGeomMap(p,x);

  PetscReal f = user->S0;
  PetscReal d = x[dim-1];
  if(d < user->d0) f += SQ(1.-SQ(d/user->d0))*(user->Sin-user->S0);

  const PetscReal *N = (typeof(N)) p->shape[0];

  PetscInt a,b;
  for(a=0; a<nen; a++){
    for(b=0; b<nen; b++) K[b*nen+a] = N[a]*N[b];
    F[a] = N[a]*f;
  }
  return 0;
}

PetscErrorCode FormInitialCondition(IGA iga,Vec U,AppCtx *user)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  Mat A;
  Vec b;
  ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&b);CHKERRQ(ierr);
  ierr = IGASetFormSystem(iga,L2Projection,user);CHKERRQ(ierr);
  ierr = IGAComputeSystem(iga,A,b);CHKERRQ(ierr);

  KSP ksp;
  ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPCG);CHKERRQ(ierr);

  ierr = KSPSetOptionsPrefix(ksp,"l2p_");CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp,1e-8,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);

  ierr = KSPSolve(ksp,b,U);CHKERRQ(ierr);

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

void CapillaryPressure(PetscScalar S,PetscScalar lambda,PetscScalar kappa,PetscScalar *J,PetscScalar *dJdS)
{
  if(J){
    *J = pow(S,-1./lambda);
    *J *= 1.-exp(kappa*(S-1.))*(S*kappa*lambda/(lambda-1.)+1.);
  }
  if(dJdS){
    *dJdS  = pow(S,-1.-1./lambda)*((S*kappa*lambda/(lambda-1.)+1.)*exp(kappa*(S-1.))-1.)/lambda;
    *dJdS += -pow(S, -1./lambda)*kappa*(S*kappa*lambda/(lambda-1.)+lambda/(lambda-1.)+1.)*exp(kappa*(S-1.));
  }
}

PetscErrorCode ZeroFluxResidual(IGAPoint p,
                                PetscReal shift,const PetscScalar *V,
                                PetscReal t,const PetscScalar *U,
                                PetscScalar *R,void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;

  PetscInt nen = p->nen;
  PetscInt dim = p->dim;

  PetscScalar S1[dim];
  IGAPointFormGrad(p,U,&S1[0]);

  const PetscReal *N0 = (typeof(N0)) p->shape[0];
  const PetscReal *n = p->normal;

  PetscInt a,i;
  for (a=0; a<nen; a++) {
    PetscScalar n_gS = 0.0;
    for(i=0;i<dim;i++) n_gS += n[i]*S1[i];
    R[a] = user->penalty*N0[a]*n_gS;
  }
  return 0;
}

PetscErrorCode Residual(IGAPoint p,
                        PetscReal shift,const PetscScalar *V,
                        PetscReal t,const PetscScalar *U,
                        PetscScalar *R,void *ctx)
{
  if (p->atboundary)
    return ZeroFluxResidual(p,shift,V,t,U,R,ctx);

  AppCtx *user = (AppCtx *)ctx;

  PetscInt a,i;
  PetscInt nen = p->nen;
  PetscInt dim = p->dim;

  PetscScalar S_t,S,delS=0;
  PetscScalar S1[dim],S2[dim][dim];
  IGAPointFormValue(p,V,&S_t);
  IGAPointFormValue(p,U,&S);
  IGAPointFormGrad (p,U,&S1[0]);
  IGAPointFormHess (p,U,&S2[0][0]);
  for(i=0;i<dim;i++) delS += S2[i][i];

  S = PetscMax(1000*PETSC_MACHINE_EPSILON,PetscRealPart(S));

  const PetscReal  *N0            = (typeof(N0)) p->shape[0];
  const PetscReal (*N1)[dim]      = (typeof(N1)) p->shape[1];
  const PetscReal (*N2)[dim][dim] = (typeof(N2)) p->shape[2];

  PetscScalar kappa     = user->kappa;
  PetscScalar lambda    = user->lambda;
  PetscScalar rNGr      = 1.0/user->NGr;
  PetscScalar kr        = pow(S,user->m);
  PetscScalar dkr_dS    = user->m*pow(S,user->m-1.);
  PetscScalar kD        = 1;
  PetscScalar dkD_dX[3] = {0,0,0};
  PetscScalar dJdS;
  CapillaryPressure(S,lambda,kappa,NULL,&dJdS);

  delS *= rNGr*rNGr*rNGr; // include the NGr^-3 into delS

  for (a=0; a<nen; a++) {
    PetscScalar gN_dot_gS=0,delN=0,gkD_dot_gN=0;
    for(i=0;i<dim;i++) {
      gN_dot_gS  += N1[a][i]*S1[i];
      delN       += N2[a][i][i];
      gkD_dot_gN += dkD_dX[i]*N1[a][i];
    }
    R[a]  = N0[a]*S_t - kD*kr*(N1[a][dim-1]+rNGr*dJdS*gN_dot_gS);
    if(user->phase_field){
      R[a] += (gkD_dot_gN*kr + kD*(dkr_dS*gN_dot_gS + kr*delN))*delS;
    }
  }
  return 0;
}

PetscErrorCode ZeroFluxJacobian(IGAPoint p,
                                PetscReal shift,const PetscScalar *V,
                                PetscReal t,const PetscScalar *U,
                                PetscScalar *J,void *ctx)
{
  return 0;
}

PetscErrorCode Jacobian(IGAPoint p,
                        PetscReal shift,const PetscScalar *V,
                        PetscReal t,const PetscScalar *U,
                        PetscScalar *J,void *ctx)
{
  if (p->atboundary)
    return ZeroFluxJacobian(p,shift,V,t,U,J,ctx);
  return 0;
}


int main(int argc, char *argv[]) {

  PetscErrorCode  ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  AppCtx user;
  user.m       = 7;
  user.kappa   = 50;
  user.lambda  = 4;
  user.NGr     = 50;
  user.S0      = 0.01;
  user.Sin     = 0.4;
  user.d0      = 0.25;
  user.penalty = 100;
  user.phase_field = PETSC_TRUE;

  PetscInt dim = 2, p = 2, k = 1, N = 256, L = 2;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","Richard's Equation Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  IGA         iga;
  IGAAxis     axis;
  PetscInt    dir,side;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDim(iga,dim);CHKERRQ(ierr);
  ierr = IGASetDof(iga,1);CHKERRQ(ierr);
  for(dir=0;dir<dim;dir++){
    ierr = IGAGetAxis(iga,dir,&axis);CHKERRQ(ierr);
    if(dir < dim-1){
      ierr = IGAAxisSetPeriodic(axis,PETSC_TRUE);CHKERRQ(ierr);
    }
    ierr = IGAAxisSetDegree(axis,p);CHKERRQ(ierr);
    ierr = IGAAxisInitUniform(axis,N,0.0,L,k);CHKERRQ(ierr);
  }
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  // Boundary conditions
  for(dir=0;dir<dim;dir++)
    for(side=0;side<2;side++){
      if(dir == dim-1){
        ierr = IGASetBoundaryForm(iga,dir,side,PETSC_TRUE);CHKERRQ(ierr);
        ierr = IGASetBoundaryForm(iga,dir,side,PETSC_TRUE);CHKERRQ(ierr);
        if(side == 0){
          ierr = IGASetBoundaryValue(iga,dir,side,0,user.Sin);CHKERRQ(ierr);
        }
      }
    }

  ierr = IGASetFormIFunction(iga,Residual,&user);CHKERRQ(ierr);
  ierr = IGASetFormIJacobian(iga,Jacobian,&user);CHKERRQ(ierr);

  TS     ts;
  ierr = IGACreateTS(iga,&ts);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,150.0);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,0.25);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSALPHA);CHKERRQ(ierr);
  ierr = TSAlphaSetRadius(ts,0.5);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  Vec       U;
  ierr = IGACreateVec(iga,&U);CHKERRQ(ierr);
  ierr = FormInitialCondition(iga,U,&user);CHKERRQ(ierr);
  ierr = TSSolve(ts,U);CHKERRQ(ierr);

  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
