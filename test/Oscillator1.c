#include <petsc.h>
#include <petscts1.h>

#if PETSC_VERSION_LE(3,3,0)
#define TSSolve(ts,x)   TSSolve(ts,x,NULL)
#define TSRegister(s,f) TSRegister(s,0,0,f)
#endif

typedef struct {
  PetscReal Omega; /* frequency */
  PetscReal Xi;    /* damping   */
} UserParams;

#undef  __FUNCT__
#define __FUNCT__ "Residual"
PetscErrorCode Residual(TS ts,PetscReal t,Vec X,Vec V,Vec R,void *ctx)
{
  UserParams *user = (UserParams *)ctx;
  PetscReal Omega = user->Omega, Xi = user->Xi;
  const PetscScalar *x,*x_t;
  PetscScalar *r;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(V,&x_t);CHKERRQ(ierr);
  ierr = VecGetArray(R,&r);CHKERRQ(ierr);

  r[0] = x_t[0] - x[1];
  r[1] = x_t[1] + (2*Xi*Omega)*x[1] + (Omega*Omega)*x[0];

  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(V,&x_t);CHKERRQ(ierr);
  ierr = VecRestoreArray(R,&r);CHKERRQ(ierr);

  ierr = VecAssemblyBegin(R);CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (R);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "Tangent"
PetscErrorCode Tangent(TS ts,PetscReal t,Vec X,Vec A,PetscReal shift,Mat J,Mat P,void *ctx)
{
  UserParams *user = (UserParams *)ctx;
  PetscReal Omega = user->Omega, Xi = user->Xi;
  PetscReal T[2][2] = {{0,0},{0,0}};
  PetscInt i,j;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  T[0][0] = shift;
  T[0][1] = -1;
  T[1][0] = Omega*Omega;
  T[1][1] = shift + 2*Xi*Omega;

  for (i=0; i<2; i++)
    for (j=0; j<2; j++)
      {ierr = MatSetValue(P,i,j,T[i][j],INSERT_VALUES);CHKERRQ(ierr);}
  ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (J != P) {
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}
#if PETSC_VERSION_LT(3,5,0)
PetscErrorCode Tangent_Legacy(TS ts,PetscReal t,Vec U,Vec V,PetscReal shift,Mat *J,Mat *P,MatStructure *m,void *ctx)
{*m = SAME_NONZERO_PATTERN; return Tangent(ts,t,U,V,shift,*J,*P,ctx);}
#define Tangent Tangent_Legacy
#endif

#undef  __FUNCT__
#define __FUNCT__ "Monitor"
PetscErrorCode Monitor(TS ts,PetscInt i,PetscReal t,Vec U,void *ctx)
{
  const char *filename = (const char *)ctx;
  static FILE *fp = 0;
  Vec X;
  const PetscScalar *x;
  TSConvergedReason reason;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if (!fp) {ierr = PetscFOpen(PETSC_COMM_SELF,filename,"w",&fp);CHKERRQ(ierr);}
  ierr = TSGetSolution(ts,&X);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"%g %g %g\n",(double)t,(double)x[0],(double)x[1]);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = TSGetConvergedReason(ts,&reason); CHKERRQ(ierr);
  if (reason) {ierr = PetscFClose(PETSC_COMM_SELF,fp);CHKERRQ(ierr); fp=0;}

  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
PetscErrorCode TSCreate_Alpha1(TS);
EXTERN_C_END

#undef  __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[]) {

  TS             ts;
  Vec            R;
  Mat            J;
  Vec            X;
  PetscScalar    *x;
  UserParams     user;
  PetscBool      out;
  char           output[PETSC_MAX_PATH_LEN] = {0};
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);
  ierr = TSRegister(TSALPHA1,TSCreate_Alpha1);CHKERRQ(ierr);

  user.Omega = 1.0;
  user.Xi    = 0.0;
  ierr = PetscOptionsBegin(PETSC_COMM_SELF,"","Oscillator1 Options","TS");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-frequency","Frequency",__FILE__,user.Omega,&user.Omega,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-damping",  "Damping",  __FILE__,user.Xi,   &user.Xi,   NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-output","Output",__FILE__,output,output,sizeof(output),&out);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (out && !output[0]) {ierr = PetscStrcpy(output,"Oscillator.out");CHKERRQ(ierr);}

  ierr = TSCreate(PETSC_COMM_SELF,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSALPHA1);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,PETSC_MAX_INT,2*M_PI * 5);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,0.01);CHKERRQ(ierr);

  ierr = VecCreateSeq(PETSC_COMM_SELF,2,&R);CHKERRQ(ierr);
  ierr = VecSetUp(R);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,2,2,NULL,&J);CHKERRQ(ierr);
  ierr = MatSetUp(J);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,R,Residual,&user);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,J,J,Tangent,&user);CHKERRQ(ierr);
  ierr = VecDestroy(&R);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);

  if (out) {ierr = TSMonitorSet(ts,Monitor,output,NULL);CHKERRQ(ierr);}

  ierr = VecCreateSeq(PETSC_COMM_SELF,2,&X);CHKERRQ(ierr);
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  x[0] = 1.0;
  x[1] = 0.0;
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);

  ierr = TSSetSolution(ts,X);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSolve(ts,X);CHKERRQ(ierr);

  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);

  return 0;
}
