#include "petiga.h"

typedef struct { 
  PetscReal lambda;
} AppCtx;

EXTERN_C_BEGIN
extern PetscErrorCode Bratu_Function(IGAPoint,const PetscScalar U[],PetscScalar F[],void *);
extern PetscErrorCode Bratu_Jacobian(IGAPoint,const PetscScalar U[],PetscScalar J[],void *);
EXTERN_C_END

EXTERN_C_BEGIN
extern PetscErrorCode Bratu_IFunction(IGAPoint,PetscReal dt,
                                      PetscReal a,const PetscScalar *V,
                                      PetscReal t,const PetscScalar *U,
                                      PetscScalar *F,void *ctx);
extern PetscErrorCode Bratu_IJacobian(IGAPoint,PetscReal dt,
                                      PetscReal a,const PetscScalar *V,
                                      PetscReal t,const PetscScalar *U,
                                      PetscScalar *F,void *ctx);
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[]) {

  PetscErrorCode  ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);
  
  
  PetscBool steady = PETSC_TRUE;
  PetscReal lambda = 6.80;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","Bratu Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-lambda","Bratu parameter",__FILE__,lambda,&lambda,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-steady","Steady problem",__FILE__,steady,&steady,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);

  ierr = IGASetDof(iga,1);CHKERRQ(ierr);
  IGABoundary bnd;
  PetscInt dir,side;
  for (dir=0; dir<2; dir++) {
    for (side=0; side<2; side++) {
      ierr = IGAGetBoundary(iga,dir,side,&bnd);CHKERRQ(ierr);
      ierr = IGABoundarySetValue(bnd,0,0.0);CHKERRQ(ierr);
    }
  }

  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);

  PetscInt dim;
  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
  if (dim < 1) {ierr = IGASetDim(iga,dim=2);CHKERRQ(ierr);}
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  AppCtx ctx;
  ctx.lambda = lambda;
  if (steady) {
    ierr = IGASetUserFunction(iga,Bratu_Function,&ctx);CHKERRQ(ierr);
    ierr = IGASetUserJacobian(iga,Bratu_Jacobian,&ctx);CHKERRQ(ierr);
  } else {
    ierr = IGASetUserIFunction(iga,Bratu_IFunction,&ctx);CHKERRQ(ierr);
    ierr = IGASetUserIJacobian(iga,Bratu_IJacobian,&ctx);CHKERRQ(ierr);
  }

  Vec x;
  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  if (steady) {
    SNES snes;
    ierr = IGACreateSNES(iga,&snes);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
    ierr = SNESSolve(snes,0,x);CHKERRQ(ierr);
    ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  } else {
    TS ts;
    ierr = IGACreateTS(iga,&ts);CHKERRQ(ierr);
    ierr = TSSetType(ts,TSTHETA);CHKERRQ(ierr);
    ierr = TSSetDuration(ts,10000,0.1);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,0.01);CHKERRQ(ierr);
    ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
    ierr = TSSolve(ts,x,PETSC_NULL);CHKERRQ(ierr);
    ierr = TSDestroy(&ts);CHKERRQ(ierr);
  }
  PetscBool draw = PETSC_FALSE;
  ierr = PetscOptionsGetBool(0,"-draw",&draw,0);CHKERRQ(ierr);
  if (draw && dim < 3) {ierr = VecView(x,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}
  ierr = VecDestroy(&x);CHKERRQ(ierr);

  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  PetscBool flag = PETSC_FALSE;
  PetscReal secs = -1;
  ierr = PetscOptionsHasName(0,"-pause",&flag);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(0,"-pause",&secs,0);CHKERRQ(ierr);
  if (flag) {ierr = PetscSleep(secs);CHKERRQ(ierr);}

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
