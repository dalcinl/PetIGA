#include <petsc.h>
#include <petscts2.h>

typedef struct {
  PetscReal Omega;   /* natural frequency */
  PetscReal Xi;      /* damping coefficient  */
  PetscReal init[2]; /* initial conditions */
} UserParams;

static void Exact(double t,
                  double omega,double xi,double u0,double v0,
                  double *ut,double *vt)
{
  double u,v;
  if (xi < 1) {
    double a  = xi*omega;
    double w  = sqrt(1-xi*xi)*omega;
    double C1 = (v0 + a*u0)/w;
    double C2 = u0;
    u = exp(-a*t) * (C1*sin(w*t) + C2*cos(w*t));
    v = (- a * exp(-a*t) * (C1*sin(w*t) + C2*cos(w*t))
         + w * exp(-a*t) * (C1*cos(w*t) - C2*sin(w*t)));
  } else if (xi > 1) {
    double w  = sqrt(xi*xi-1)*omega;
    double C1 = (w*u0 + xi*u0 + v0)/(2*w);
    double C2 = (w*u0 - xi*u0 - v0)/(2*w);
    u = C1*exp((-xi+w)*t) + C2*exp((-xi-w)*t);
    v = C1*(-xi+w)*exp((-xi+w)*t) + C2*(-xi-w)*exp((-xi-w)*t);
  } else {
    double a  = xi*omega;
    double C1 = v0 + a*u0;
    double C2 = u0;
    u = (C1*t + C2) * exp(-a*t);
    v = (C1 - a*(C1*t + C2)) * exp(-a*t);
  }
  if (ut) *ut = u;
  if (vt) *vt = v;
}

#undef  __FUNCT__
#define __FUNCT__ "Solution"
PetscErrorCode Solution(TS ts,PetscReal t,Vec X,void *ctx)
{
  UserParams     *user = (UserParams*)ctx;
  double         Omega = (double)user->Omega;
  double         Xi    = (double)user->Xi;
  double         u,u0  = (double)user->init[0];
  double         v,v0  = (double)user->init[1];
  PetscScalar    *x;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  Exact(t,Omega,Xi,u0,v0,&u,&v);
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  x[0] = u;
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "Residual1"
PetscErrorCode Residual1(TS ts,PetscReal t,Vec X,Vec A,Vec R,void *ctx)
{
  UserParams        *user = (UserParams*)ctx;
  PetscReal         Omega = user->Omega;
  const PetscScalar *x,*a;
  PetscScalar       *r;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(A,&a);CHKERRQ(ierr);
  ierr = VecGetArray(R,&r);CHKERRQ(ierr);

  r[0] = a[0] + (Omega*Omega)*x[0];

  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(A,&a);CHKERRQ(ierr);
  ierr = VecRestoreArray(R,&r);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(R);CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (R);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "Tangent1"
PetscErrorCode Tangent1(TS ts,PetscReal t,Vec X,Vec A,PetscReal shiftA,Mat J,Mat P,void *ctx)
{
  UserParams     *user = (UserParams*)ctx;
  PetscReal      Omega = user->Omega;
  PetscReal      T = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  T = shiftA + (Omega*Omega);

  ierr = MatSetValue(P,0,0,T,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (J != P) {
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#if PETSC_VERSION_LT(3,5,0)
PetscErrorCode Tangent1_Legacy(TS ts,PetscReal t,Vec U,Vec V,PetscReal shift,Mat *J,Mat *P,MatStructure *m,void *ctx)
{*m = SAME_NONZERO_PATTERN; return Tangent1(ts,t,U,V,shift,*J,*P,ctx);}
#define Tangent1 Tangent1_Legacy
#endif

#undef  __FUNCT__
#define __FUNCT__ "Residual2"
PetscErrorCode Residual2(TS ts,PetscReal t,Vec X,Vec V,Vec A,Vec R,void *ctx)
{
  UserParams         *user = (UserParams*)ctx;
  PetscReal          Omega = user->Omega, Xi = user->Xi;
  const PetscScalar *x,*v,*a;
  PetscScalar       *r;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(V,&v);CHKERRQ(ierr);
  ierr = VecGetArrayRead(A,&a);CHKERRQ(ierr);
  ierr = VecGetArray(R,&r);CHKERRQ(ierr);

  r[0] = a[0] + (2*Xi*Omega)*v[0] + (Omega*Omega)*x[0];

  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(V,&v);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(A,&a);CHKERRQ(ierr);
  ierr = VecRestoreArray(R,&r);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(R);CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (R);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "Tangent2"
PetscErrorCode Tangent2(TS ts,PetscReal t,Vec X,Vec V,Vec A,PetscReal shiftV,PetscReal shiftA,Mat J,Mat P,void *ctx)
{
  UserParams     *user = (UserParams*)ctx;
  PetscReal      Omega = user->Omega, Xi = user->Xi;
  PetscReal      T = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  T = shiftA + shiftV * (2*Xi*Omega) + (Omega*Omega);

  ierr = MatSetValue(P,0,0,T,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (J != P) {
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "Monitor"
PetscErrorCode Monitor(TS ts,PetscInt i,PetscReal t,Vec U,void *ctx)
{
  const char        *filename = (const char*)ctx;
  static FILE       *fp = NULL;
  Vec               X,V;
  const PetscScalar *x,*v;
  TSConvergedReason reason;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (!fp) {ierr = PetscFOpen(PETSC_COMM_SELF,filename,"w",&fp);CHKERRQ(ierr);}
  ierr = TSGetSolution2(ts,&X,&V);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(V,&v);CHKERRQ(ierr);
  ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"%g %g %g\n",(double)t,(double)PetscRealPart(x[0]),(double)PetscRealPart(v[0]));CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(V,&v);CHKERRQ(ierr);
  ierr = TSGetConvergedReason(ts,&reason);CHKERRQ(ierr);
  if (reason) {ierr = PetscFClose(PETSC_COMM_SELF,fp);CHKERRQ(ierr); fp = NULL;}
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
extern PetscErrorCode TSCreate_Alpha2(TS);
EXTERN_C_END

#undef  __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  TS             ts;
  Vec            R;
  Mat            J;
  Vec            X,V;
  PetscScalar    *x,*v;
  PetscInt       n = 2;
  UserParams     user = {/*Omega=*/ 1, /*Xi=*/ 0, /*u0,v0=*/ {1, 0}};
  PetscBool      out;
  char           output[PETSC_MAX_PATH_LEN] = {0};
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);
  ierr = TSRegister(TSALPHA2,TSCreate_Alpha2);CHKERRQ(ierr);

  ierr = PetscOptionsBegin(PETSC_COMM_SELF,"","Oscillator2 Options","TS");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-frequency","Frequency",__FILE__,user.Omega,&user.Omega,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-damping","Damping",__FILE__,user.Xi,&user.Xi,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsRealArray("-initial","Initial conditions",__FILE__,user.init,&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-output","Output filename",__FILE__,output,output,sizeof(output),&out);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (out && !output[0]) {ierr = PetscStrcpy(output,"Oscillator.out");CHKERRQ(ierr);}

  ierr = TSCreate(PETSC_COMM_SELF,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSALPHA2);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,PETSC_MAX_INT,5*(2*PETSC_PI));CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,0.01);CHKERRQ(ierr);
  if (out) {ierr = TSMonitorSet(ts,Monitor,output,NULL);CHKERRQ(ierr);}

  ierr = VecCreateSeq(PETSC_COMM_SELF,1,&R);CHKERRQ(ierr);
  ierr = VecSetUp(R);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,1,1,NULL,&J);CHKERRQ(ierr);
  ierr = MatSetUp(J);CHKERRQ(ierr);
  if (user.Xi) {
    ierr = TSSetIFunction2(ts,R,Residual2,&user);CHKERRQ(ierr);
    ierr = TSSetIJacobian2(ts,J,J,Tangent2,&user);CHKERRQ(ierr);
  } else {
    ierr = TSSetIFunction(ts,R,Residual1,&user);CHKERRQ(ierr);
    ierr = TSSetIJacobian(ts,J,J,Tangent1,&user);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&R);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = TSSetSolutionFunction(ts,Solution,&user);CHKERRQ(ierr);

  ierr = VecCreateSeq(PETSC_COMM_SELF,1,&X);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,1,&V);CHKERRQ(ierr);
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(V,&v);CHKERRQ(ierr);
  x[0] = user.init[0];
  v[0] = user.init[1];
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
