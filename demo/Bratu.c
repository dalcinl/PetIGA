/*
   This code solves the steady and unsteady Bratu equation. It also
   demonstrates how the user-specified routines, here the Function and
   Jacobian routines, can be implemented in Fortran (see BratuFJ.F90)
   yet called from PetIGA in C.

   keywords: steady, transient, scalar, implicit, nonlinear, testing,
   dimension independent, collocation, fortran
*/
#include "petiga.h"

typedef struct {
  PetscReal lambda;
} AppCtx;

EXTERN_C_BEGIN
extern PetscErrorCode Bratu_Function(IGAPoint,const PetscScalar U[],PetscScalar F[],void *ctx);
extern PetscErrorCode Bratu_Jacobian(IGAPoint,const PetscScalar U[],PetscScalar J[],void *ctx);
EXTERN_C_END

EXTERN_C_BEGIN
extern PetscErrorCode Bratu_IFunction(IGAPoint,
                                      PetscReal a,const PetscScalar *V,
                                      PetscReal t,const PetscScalar *U,
                                      PetscScalar *F,void *ctx);
extern PetscErrorCode Bratu_IJacobian(IGAPoint,
                                      PetscReal a,const PetscScalar *V,
                                      PetscReal t,const PetscScalar *U,
                                      PetscScalar *F,void *ctx);
EXTERN_C_END

int main(int argc, char *argv[]) {

  PetscErrorCode  ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);


  PetscBool fd = PETSC_FALSE;
  PetscBool steady = PETSC_TRUE;
  PetscReal lambda = 6.80;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","Bratu Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-steady","Steady problem",__FILE__,steady,&steady,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-lambda","Bratu parameter",__FILE__,lambda,&lambda,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-iga_fd","Use FD Jacobian",__FILE__,fd,&fd,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDof(iga,1);CHKERRQ(ierr);
  PetscInt dim,dir,side;
  for (dir=0; dir<3; dir++) {
    for (side=0; side<2; side++) {
      PetscInt    field = 0;
      PetscScalar value = 0.0;
      ierr = IGASetBoundaryValue(iga,dir,side,field,value);CHKERRQ(ierr);
    }
  }
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
  if (dim < 1) {ierr = IGASetDim(iga,dim=2);CHKERRQ(ierr);}
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  AppCtx ctx;
  ctx.lambda = lambda;
  if (steady) {
    ierr = IGASetFormFunction(iga,Bratu_Function,&ctx);CHKERRQ(ierr);
    ierr = IGASetFormJacobian(iga,fd?IGAFormJacobianFD:Bratu_Jacobian,&ctx);CHKERRQ(ierr);
  } else {
    ierr = IGASetFormIFunction(iga,Bratu_IFunction,&ctx);CHKERRQ(ierr);
    ierr = IGASetFormIJacobian(iga,fd?IGAFormIJacobianFD:Bratu_IJacobian,&ctx);CHKERRQ(ierr);
  }

  Vec x;
  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);

  if (steady) {
    SNES snes;
    ierr = IGACreateSNES(iga,&snes);CHKERRQ(ierr);
    if (!iga->collocation) {
      Mat mat; KSP ksp;
      ierr = SNESGetJacobian(snes,NULL,&mat,NULL,NULL);CHKERRQ(ierr);
      ierr = MatSetOption(mat,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
      ierr = MatSetOption(mat,MAT_SPD,PETSC_TRUE);CHKERRQ(ierr);
      ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
      ierr = KSPSetType(ksp,KSPCG);CHKERRQ(ierr);
    }
    ierr = SNESSetTolerances(snes,PETSC_DEFAULT,1e-5,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
    ierr = SNESSolve(snes,NULL,x);CHKERRQ(ierr);
    ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  } else {
    TS   ts;
    SNES snes;
    ierr = IGACreateTS(iga,&ts);CHKERRQ(ierr);
    ierr = TSSetType(ts,TSTHETA);CHKERRQ(ierr);
    ierr = TSSetMaxTime(ts,0.1);CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,0.01);CHKERRQ(ierr);
    ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
    ierr = SNESSetTolerances(snes,PETSC_DEFAULT,1e-5,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);
    if (!iga->collocation) {
      Mat mat; KSP ksp;
      ierr = TSGetIJacobian(ts,NULL,&mat,NULL,NULL);CHKERRQ(ierr);
      ierr = MatSetOption(mat,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
      ierr = MatSetOption(mat,MAT_SPD,PETSC_TRUE);CHKERRQ(ierr);
      ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
      ierr = KSPSetType(ksp,KSPCG);CHKERRQ(ierr);
    }
    ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
    ierr = TSSolve(ts,x);CHKERRQ(ierr);
    ierr = TSDestroy(&ts);CHKERRQ(ierr);
  }

  PetscBool save = IGAGetOptBool(NULL,"-save",PETSC_FALSE);
  if (save) {ierr = IGAWrite(iga,"Bratu-geometry.dat");CHKERRQ(ierr);}
  if (save) {ierr = IGAWriteVec(iga,x,"Bratu-solution.dat");CHKERRQ(ierr);}

  PetscBool draw = IGAGetOptBool(NULL,"-draw",PETSC_FALSE);
  if (draw&&dim<3) {ierr = IGADrawVec(iga,x,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
