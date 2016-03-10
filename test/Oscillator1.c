#include <petsc.h>
#include <petscts1.h>

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
  x[0] = u; x[1] = v;
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "Residual"
PetscErrorCode Residual(TS ts,PetscReal t,Vec X,Vec V,Vec R,void *ctx)
{
  UserParams        *user = (UserParams*)ctx;
  PetscReal         Omega = user->Omega, Xi = user->Xi;
  const PetscScalar *x,*x_t;
  PetscScalar       *r;
  PetscErrorCode    ierr;

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
  UserParams     *user = (UserParams*)ctx;
  PetscReal      Omega = user->Omega, Xi = user->Xi;
  PetscReal      T[2][2] = {{0,0},{0,0}};
  PetscInt       i,j;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  T[0][0] = shift;       T[0][1] = -1;
  T[1][0] = Omega*Omega; T[1][1] = shift + 2*Xi*Omega;

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
  const char        *filename = (const char*)ctx;
  static FILE       *fp = NULL;
  Vec               X;
  const PetscScalar *x;
  TSConvergedReason reason;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (!fp) {ierr = PetscFOpen(PETSC_COMM_SELF,filename,"w",&fp);CHKERRQ(ierr);}
  ierr = TSGetSolution(ts,&X);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"%g %g %g\n",(double)t,(double)PetscRealPart(x[0]),(double)PetscRealPart(x[1]));CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = TSGetConvergedReason(ts,&reason);CHKERRQ(ierr);
  if (reason) {ierr = PetscFClose(PETSC_COMM_SELF,fp);CHKERRQ(ierr); fp = NULL;}
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
extern PetscErrorCode TSCreate_Alpha1(TS);
extern PetscErrorCode TSCreate_BDF(TS);
EXTERN_C_END

#undef  __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  TS             ts;
  SNES           snes;
  KSP            ksp;
  PC             pc;
  Vec            R;
  Mat            J;
  Vec            X;
  PetscScalar    *x;
  PetscInt       n = 2;
  UserParams     user = {/*Omega=*/ 1, /*Xi=*/ 0, /*u0,v0=*/ {1, 0}};
  PetscBool      out;
  char           output[PETSC_MAX_PATH_LEN] = {0};
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);
  ierr = TSRegister(TSALPHA1,TSCreate_Alpha1);CHKERRQ(ierr);
  ierr = TSRegister(TSBDF,TSCreate_BDF);CHKERRQ(ierr);

  ierr = PetscOptionsBegin(PETSC_COMM_SELF,"","Oscillator1 Options","TS");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-frequency","Frequency",__FILE__,user.Omega,&user.Omega,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-damping","Damping",__FILE__,user.Xi,&user.Xi,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsRealArray("-initial","Initial conditions",__FILE__,user.init,&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-output","Output filename",__FILE__,output,output,sizeof(output),&out);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (out && !output[0]) {ierr = PetscStrcpy(output,"Oscillator.out");CHKERRQ(ierr);}

  ierr = TSCreate(PETSC_COMM_SELF,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSALPHA1);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,PETSC_MAX_INT,5*(2*PETSC_PI));CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,0.01);CHKERRQ(ierr);
  if (out) {ierr = TSMonitorSet(ts,Monitor,output,NULL);CHKERRQ(ierr);}

  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESSetType(snes,SNESKSPONLY);CHKERRQ(ierr);
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);

  ierr = VecCreateSeq(PETSC_COMM_SELF,2,&R);CHKERRQ(ierr);
  ierr = VecSetUp(R);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,2,2,NULL,&J);CHKERRQ(ierr);
  ierr = MatSetUp(J);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,R,Residual,&user);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,J,J,Tangent,&user);CHKERRQ(ierr);
  ierr = VecDestroy(&R);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = TSSetSolutionFunction(ts,Solution,&user);CHKERRQ(ierr);

  ierr = VecCreateSeq(PETSC_COMM_SELF,2,&X);CHKERRQ(ierr);
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  x[0] = user.init[0];
  x[1] = user.init[1];
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);

  ierr = TSSetSolution(ts,X);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSolve(ts,X);CHKERRQ(ierr);

  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);

  return 0;
}
