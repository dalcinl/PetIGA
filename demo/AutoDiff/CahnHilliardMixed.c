#include <petiga.h>

typedef struct {
  PetscReal theta;
  PetscReal alpha;
  PetscReal cbar;
} Params;

EXTERN_C_BEGIN
PetscErrorCode IFunctionC99(IGAPoint  q,
                            PetscReal a,const PetscScalar V[],
                            PetscReal t,const PetscScalar U[],
                            PetscScalar F[],void *ctx);
PetscErrorCode IJacobianC99(IGAPoint  q,
                            PetscReal a,const PetscScalar V[],
                            PetscReal t,const PetscScalar U[],
                            PetscScalar J[],void *ctx);
EXTERN_C_END

EXTERN_C_BEGIN
PetscErrorCode IFunctionCXX(IGAPoint  q,
                            PetscReal a,const PetscScalar V[],
                            PetscReal t,const PetscScalar U[],
                            PetscScalar F[],void *ctx);
PetscErrorCode IJacobianCXX(IGAPoint  q,
                            PetscReal a,const PetscScalar V[],
                            PetscReal t,const PetscScalar U[],
                            PetscScalar J[],void *ctx);
EXTERN_C_END

EXTERN_C_BEGIN
PetscErrorCode IFunctionFAD(IGAPoint  q,
                            PetscReal a,const PetscScalar V[],
                            PetscReal t,const PetscScalar U[],
                            PetscScalar F[],void *ctx);
PetscErrorCode IJacobianFAD(IGAPoint  q,
                            PetscReal a,const PetscScalar V[],
                            PetscReal t,const PetscScalar U[],
                            PetscScalar J[],void *ctx);
EXTERN_C_END

PetscErrorCode FormInitial(IGA iga,Vec C,Params *user)
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

int main(int argc, char *argv[]) {

  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  /* Define simulation specific parameters */
  Params params;
  params.alpha = 3000.0; /* interface thickess parameter */
  params.theta = 1.5;    /* temperature/critical temperature */
  params.cbar  = 0.63;   /* average concentration */

  /* Set discretization options */
  char      initial[PETSC_MAX_PATH_LEN] = {0};
  PetscBool output  = PETSC_FALSE;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","CahnHilliard Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsString("-initial","Load initial solution from file",__FILE__,initial,initial,sizeof(initial),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-output","Enable output files",__FILE__,output,&output,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-cbar","Initial average concentration",__FILE__,params.cbar,&params.cbar,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-alpha","Interface thickess parameter",__FILE__,params.alpha,&params.alpha,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-theta","Ratio temperature/critical temperature",__FILE__,params.theta,&params.theta,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = IGAOptionsAlias("-wrap",  "", "-iga_periodic");CHKERRQ(ierr);
  ierr = IGAOptionsAlias("-dim",  "2", "-iga_dim");CHKERRQ(ierr);
  ierr = IGAOptionsAlias("-deg",  "2", "-iga_degree");CHKERRQ(ierr);
  ierr = IGAOptionsAlias("-nel", "64", "-iga_elements");CHKERRQ(ierr);
  ierr = IGAOptionsAlias("-k",   NULL, "-iga_continuity");CHKERRQ(ierr);

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDof(iga,2);CHKERRQ(ierr);
  ierr = IGASetOrder(iga,1);CHKERRQ(ierr);
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  ierr = IGASetFieldName(iga,0,"c");CHKERRQ(ierr);
  ierr = IGASetFieldName(iga,1,"sigma");CHKERRQ(ierr);

  PetscBool c99 = IGAGetOptBool(NULL,"-c99",PETSC_TRUE);
  if (c99) {
    IGASetFormIFunction(iga,IFunctionC99,&params);
    IGASetFormIJacobian(iga,IJacobianC99,&params);
  }
  //PetscBool f90 = IGAGetOptBool(NULL,"-f90",PETSC_FALSE);
  //if (f90) {
  //  IGASetFormIFunction(iga,IFunctionF90,&params);
  //  IGASetFormIJacobian(iga,IJacobianF90,&params);
  //}
  PetscBool cxx = IGAGetOptBool(NULL,"-cxx",PETSC_FALSE);
  if (cxx) {
    IGASetFormIFunction(iga,IFunctionCXX,&params);
    IGASetFormIJacobian(iga,IJacobianCXX,&params);
  }
  PetscBool fad = IGAGetOptBool(NULL,"-fad",PETSC_FALSE);
  if (fad) {
    IGASetFormIFunction(iga,IFunctionFAD,&params);
    IGASetFormIJacobian(iga,IJacobianFAD,&params);
  }
  PetscBool fd = IGAGetOptBool(NULL,"-fd",PETSC_FALSE);
  if (fd) {IGASetFormIJacobian(iga,IGAFormIJacobianFD,&params);}

  TS ts;
  ierr = IGACreateTS(iga,&ts);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,1.0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,1e-11);CHKERRQ(ierr);

  ierr = TSSetType(ts,TSALPHA);CHKERRQ(ierr);
  ierr = TSAlphaSetRadius(ts,0.5);CHKERRQ(ierr);
  ierr = TSAlphaUseAdapt(ts,PETSC_TRUE);CHKERRQ(ierr);
  ierr = TSSetMaxSNESFailures(ts,-1);CHKERRQ(ierr);
  {
    PetscInt  i;
    Vec       vatol,vrtol;
    PetscReal atol[] = {PETSC_INFINITY, PETSC_INFINITY};
    PetscReal rtol[] = {PETSC_INFINITY, PETSC_INFINITY};
    ierr = IGACreateVec(iga,&vatol);CHKERRQ(ierr);
    ierr = IGACreateVec(iga,&vrtol);CHKERRQ(ierr);
    ierr = TSGetTolerances(ts,&atol[0],NULL,&rtol[0],NULL);CHKERRQ(ierr);
    for (i=0; i<2; i++) {
      ierr = VecStrideSet(vatol,i,atol[i]);CHKERRQ(ierr);
      ierr = VecStrideSet(vrtol,i,rtol[i]);CHKERRQ(ierr);
    }
    ierr = TSSetTolerances(ts,atol[0],vatol,rtol[0],vrtol);CHKERRQ(ierr);
    ierr = VecDestroy(&vatol);CHKERRQ(ierr);
    ierr = VecDestroy(&vrtol);CHKERRQ(ierr);
  }

  if (output)  {ierr = TSMonitorSet(ts,OutputMonitor,&params,NULL);CHKERRQ(ierr);}
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  Vec C;
  ierr = TSGetSolution(ts,&C);CHKERRQ(ierr);
  if (initial[0] == 0) { /* initial condition is random */
    ierr = FormInitial(iga,C,&params);CHKERRQ(ierr);
  } else {               /* initial condition from datafile */
    ierr = IGAReadVec(iga,C,initial);CHKERRQ(ierr);
  }
  ierr = TSSolve(ts,C);CHKERRQ(ierr);

  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
