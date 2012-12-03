#include <petsc.h>
#include <petscts2.h>

typedef struct {
  PetscReal Omega; /* frequency */
  PetscReal Xi;    /* damping   */
} UserParams;

#undef  __FUNCT__
#define __FUNCT__ "Residual1"
PetscErrorCode Residual1(TS ts,PetscReal t,Vec X,Vec A,Vec R,void *ctx)
{
  UserParams *user = (UserParams *)ctx;
  PetscReal Omega = user->Omega;
  PetscScalar *x,*a,*r;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(A,&a);CHKERRQ(ierr);
  ierr = VecGetArray(R,&r);CHKERRQ(ierr);

  r[0] = a[0] + (Omega*Omega)*x[0];

  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(A,&a);CHKERRQ(ierr);
  ierr = VecRestoreArray(R,&r);CHKERRQ(ierr);

  ierr = VecAssemblyBegin(R);CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (R);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "Tangent1"
PetscErrorCode Tangent1(TS ts,PetscReal t,Vec X,Vec A,PetscReal shiftA,Mat *J,Mat *P,MatStructure *m,void *ctx)
{
  UserParams *user = (UserParams *)ctx;
  PetscReal Omega = user->Omega;
  PetscReal T = 0;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  T = shiftA + (Omega*Omega);

  ierr = MatSetValue(*P,0,0,T,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (*P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (*J != * P) {
    ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  *m = SAME_NONZERO_PATTERN;

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "Residual2"
PetscErrorCode Residual2(TS ts,PetscReal t,Vec X,Vec V,Vec A,Vec R,void *ctx)
{
  UserParams *user = (UserParams *)ctx;
  PetscReal Omega = user->Omega, Xi = user->Xi;
  PetscScalar *x,*v,*a,*r;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(V,&v);CHKERRQ(ierr);
  ierr = VecGetArray(A,&a);CHKERRQ(ierr);
  ierr = VecGetArray(R,&r);CHKERRQ(ierr);

  r[0] = a[0] + (2*Xi*Omega)*v[0] + (Omega*Omega)*x[0];

  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(V,&v);CHKERRQ(ierr);
  ierr = VecRestoreArray(A,&a);CHKERRQ(ierr);
  ierr = VecRestoreArray(R,&r);CHKERRQ(ierr);

  ierr = VecAssemblyBegin(R);CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (R);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "Tangent2"
PetscErrorCode Tangent2(TS ts,PetscReal t,Vec X,Vec V,Vec A,PetscReal shiftV,PetscReal shiftA,Mat *J,Mat *P,MatStructure *m,void *ctx)
{
  UserParams *user = (UserParams *)ctx;
  PetscReal Omega = user->Omega, Xi = user->Xi;
  PetscReal T = 0;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  T = shiftA + shiftV * (2*Xi*Omega) + (Omega*Omega);

  ierr = MatSetValue(*P,0,0,T,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (*P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (*J != * P) {
    ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  *m = SAME_NONZERO_PATTERN;

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "Monitor"
PetscErrorCode Monitor(TS ts,PetscInt i,PetscReal t,Vec U,void *ctx)
{
  const char *filename = (const char *)ctx;
  static FILE *fp = 0;
  Vec X,V;
  const PetscScalar *x,*v;
  TSConvergedReason reason;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  if (!fp) {ierr = PetscFOpen(PETSC_COMM_SELF,filename,"w",&fp);CHKERRQ(ierr);}
  ierr = TSGetSolution2(ts,&X,&V);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(V,&v);CHKERRQ(ierr);
  ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"%G %G %G\n",t,x[0],v[0]);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(V,&v);CHKERRQ(ierr);
  ierr = TSGetConvergedReason(ts,&reason); CHKERRQ(ierr);
  if (reason) {ierr = PetscFClose(PETSC_COMM_SELF,fp);CHKERRQ(ierr); fp=0;}

  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
PetscErrorCode TSCreate_Alpha2(TS);
EXTERN_C_END

#undef  __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[]) {

  TS             ts;
  Vec            R;
  Mat            J;
  Vec            X,V;
  PetscScalar    *x,*v;
  UserParams     user;
  PetscBool      out;
  char           output[PETSC_MAX_PATH_LEN] = {0};
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);
  ierr = TSRegisterDynamic(TSALPHA2,0,"TSCreate_Alpha2",TSCreate_Alpha2);CHKERRQ(ierr);

  user.Omega = 1.0;
  user.Xi    = 0.0;
  ierr = PetscOptionsBegin(PETSC_COMM_SELF,"","Oscillator Options","TS");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-frequency","Frequency",__FILE__,user.Omega,&user.Omega,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-damping",  "Damping",  __FILE__,user.Xi,   &user.Xi,   PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-output","Output",__FILE__,output,output,sizeof(output),&out);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (out && !output[0]) {ierr = PetscStrcpy(output,"Oscillator.out");CHKERRQ(ierr);}

  ierr = TSCreate(PETSC_COMM_SELF,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSALPHA2);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,PETSC_MAX_INT,2*M_PI * 5);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,0.01);CHKERRQ(ierr);

  ierr = VecCreateSeq(PETSC_COMM_SELF,1,&R);CHKERRQ(ierr);
  ierr = VecSetUp(R);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,1,1,PETSC_NULL,&J);CHKERRQ(ierr);
  ierr = MatSetUp(J);CHKERRQ(ierr);
  if (user.Xi <= 0.0) {
    ierr = TSSetIFunction(ts,R,Residual1,&user);CHKERRQ(ierr);
    ierr = TSSetIJacobian(ts,J,J,Tangent1,&user);CHKERRQ(ierr);
  } else {
    ierr = TSSetIFunction2(ts,R,Residual2,&user);CHKERRQ(ierr);
    ierr = TSSetIJacobian2(ts,J,J,Tangent2,&user);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&R);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);

  if (output[0]) {
    ierr = TSMonitorSet(ts,Monitor,output,PETSC_NULL);CHKERRQ(ierr);
  }

  ierr = VecCreateSeq(PETSC_COMM_SELF,1,&X);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,1,&V);CHKERRQ(ierr);
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(V,&v);CHKERRQ(ierr);
  x[0] = 1.0;
  v[0] = 0.0;
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(V,&v);CHKERRQ(ierr);

  ierr = TSSetSolution2(ts,X,V);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSolve2(ts,X,V);CHKERRQ(ierr);

  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&V);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);

  return 0;
}
