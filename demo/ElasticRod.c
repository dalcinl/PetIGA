#include "petiga.h"

EXTERN_C_BEGIN

typedef struct {
  PetscReal rho; /* density */
  PetscReal E;   /* Young modulus */
} UserParams;

extern PetscErrorCode ElasticRod_IFunction(IGAPoint,
                                           PetscReal a,const PetscScalar *A,
                                           PetscReal v,const PetscScalar *V,
                                           PetscReal t,const PetscScalar *U,
                                           PetscScalar *F,void *ctx);
extern PetscErrorCode ElasticRod_IJacobian(IGAPoint,
                                           PetscReal a,const PetscScalar *A,
                                           PetscReal v,const PetscScalar *V,
                                           PetscReal t,const PetscScalar *U,
                                           PetscScalar *J,void *ctx);
EXTERN_C_END

int main(int argc, char *argv[]) {

  PetscErrorCode  ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  UserParams user;
  user.rho = 1.0;
  user.E   = 1.0;

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","ElasticRod Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-density","Density",__FILE__,user.rho,&user.rho,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-Young_modulus","Young modulus",__FILE__,user.E,&user.E,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDim(iga,1);CHKERRQ(ierr);
  ierr = IGASetDof(iga,1);CHKERRQ(ierr);

  IGAAxis axis;
  ierr = IGAGetAxis(iga,0,&axis);CHKERRQ(ierr);
  PetscInt  p = 2;            /* degree     */
  PetscInt  C = PETSC_DECIDE; /* continuity */
  PetscInt  N = 64;           /* elements   */
  PetscReal L = 1.0;          /* lenght     */
  ierr = IGAAxisSetDegree(axis,p);CHKERRQ(ierr);
  ierr = IGAAxisInitUniform(axis,N,0.0,L,C);CHKERRQ(ierr);

  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  /* Boundary conditions, U[left] = U[right] = 0 */
  PetscScalar U_left = 0.0, U_right = 0.0;
  /* left,  (dir=0, side=0) */
  ierr = IGASetBoundaryValue(iga,0,0,0,U_left);CHKERRQ(ierr);
  /* right  (dir=0, side=1) */
  ierr = IGASetBoundaryValue(iga,0,1,0,U_right);CHKERRQ(ierr);

  /* Residual and Tangent user routines */
  ierr = IGASetFormI2Function(iga,ElasticRod_IFunction,&user);CHKERRQ(ierr);
  ierr = IGASetFormI2Jacobian(iga,ElasticRod_IJacobian,&user);CHKERRQ(ierr);

  /* Timestepper, t_final=5.0, delta_t = 0.01 */
  TS ts;
  ierr = IGACreateTS2(iga,&ts);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,5.0);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,0.01);CHKERRQ(ierr);
  ierr = TSAlpha2SetRadius(ts,0.5);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* Initial conditions, U[center] = 1.0, V[:] = 0.0 */
  Vec U,V; PetscInt n;
  ierr = TS2GetSolution(ts,&U,&V);CHKERRQ(ierr);
  ierr = VecGetSize(U,&n);CHKERRQ(ierr);
  ierr = VecSetValue(U,n/2,1.0,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(U);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(U);CHKERRQ(ierr);

  /* Solve */
  ierr = TSSolve(ts,U);CHKERRQ(ierr);
  if (0) { /* write final solution */
    ierr = IGAWriteVec(iga,U,"ElasticRod_U.dat");CHKERRQ(ierr);
    ierr = IGAWriteVec(iga,V,"ElasticRod_V.dat");CHKERRQ(ierr);
  }

  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
