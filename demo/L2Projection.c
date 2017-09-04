#include "petiga.h"

static PetscScalar Linear(PetscInt dim,PetscReal x[3])
{
  PetscInt i; double f = 0;
  for (i=0; i<dim; i++) f += x[i];
  return (PetscScalar)f;
}

static PetscScalar Quadratic(PetscInt dim,PetscReal x[3])
{
  PetscInt i; double f = 0;
  for (i=0; i<dim; i++) f += x[i]*x[i];
  return (PetscScalar)f;
}

static PetscScalar Cubic(PetscInt dim,PetscReal x[3])
{
  PetscInt i; double f = 0;
  for (i=0; i<dim; i++) f += x[i]*x[i]*x[i];
  return (PetscScalar)f;
}

static PetscScalar Quartic(PetscInt dim,PetscReal x[3])
{
  PetscInt i; double f = 0;
  for (i=0; i<dim; i++) f += x[i]*x[i]*x[i]*x[i];
  return (PetscScalar)f;
}

static PetscScalar Hill(PetscInt dim,PetscReal xyz[3])
{
  double x = (double)xyz[0]; x = 2.5*x+1;
  double y = (double)xyz[1]; y = 2.0*y+0;
  double f = exp(-x*x-y*y) + 0.5 * exp(-(x-2)*(x-2)-(y-0.5)*(y-0.5));
  return (PetscScalar)f;
}

static PetscScalar Peaks(PetscInt dim,PetscReal xyz[3])
{
  double x = (double)xyz[0]*3;
  double y = (double)xyz[1]*3;
  double f = 3 * pow(1-x,2) * exp(-pow(x,2) - pow(y+1,2))
             - 10 * (x/5 - pow(x,3) - pow(y,5)) * exp(-pow(x,2) - pow(y,2))
             - 1.0/3 * exp(-pow(x+1,2) - pow(y,2));
  return (PetscScalar)f;
}

static PetscScalar Sine(PetscInt dim,PetscReal x[3])
{
  PetscInt i; double f = 1;
  for (i=0; i<dim; i++) f *= sin(M_PI*x[i]);
  return (PetscScalar)f;
}

static PetscScalar Step(PetscInt dim,PetscReal x[3])
{
  PetscInt i; double f = 0;
  for (i=0; i<dim; i++) f += (x[i] < 0.0) ? -1.0: +1.0;
  return (PetscScalar)f;
}

typedef struct {
  PetscScalar (*Function)(PetscInt dim,PetscReal xyz[3]);
} AppCtx;

PetscErrorCode System(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  AppCtx  *app = (AppCtx*)ctx;
  PetscInt nen = p->nen;
  PetscInt dim = p->dim;

  PetscReal xyz[3] = {0,0,0};
  IGAPointFormGeomMap(p,xyz);
  PetscScalar f = app->Function(dim,xyz);

  const PetscReal *N = (typeof(N)) p->shape[0];

  PetscInt a,b;
  for (a=0; a<nen; a++) {
    for (b=0; b<nen; b++) {
      K[a*nen+b] = N[a] * N[b];
    }
    F[a] = N[a] * f;
  }

  return 0;
}

PetscErrorCode Exact(IGAPoint p,PetscInt order,PetscScalar value[],void *ctx)
{
  AppCtx  *app = (AppCtx*)ctx;
  PetscInt dim = p->dim;
  PetscReal xyz[3] = {0,0,0};
  IGAPointFormGeomMap(p,xyz);
  value[0] = app->Function(dim,xyz);
  return 0;
}

int main(int argc, char *argv[]) {

  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  const char *choicelist[] = {"linear", "quadratic", "cubic", "quartic", "hill", "peaks", "sine", "step"};
  PetscInt choice = 0, nchoices = (PetscInt)(sizeof(choicelist)/sizeof(choicelist[0]));
  PetscBool print_error = PETSC_FALSE;
  PetscBool check_error = PETSC_FALSE;
  PetscBool save = PETSC_FALSE;
  PetscBool draw = PETSC_FALSE;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","L2Projection Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsEList("-function","Function to project",__FILE__,choicelist,nchoices,choicelist[choice],&choice,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-print_error","Prints the L2 error of the solution",__FILE__,print_error,&print_error,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-check_error","Checks the L2 error of the solution",__FILE__,check_error,&check_error,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-save","Save the solution to file",__FILE__,save,&save,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-draw","Draw the solution to the screen",__FILE__,draw,&draw,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  AppCtx app;
  switch (choice) {
  case 0: app.Function = Linear;    break;
  case 1: app.Function = Quadratic; break;
  case 2: app.Function = Cubic;     break;
  case 3: app.Function = Quartic;   break;
  case 4: app.Function = Hill;      break;
  case 5: app.Function = Peaks;     break;
  case 6: app.Function = Sine;      break;
  case 7: app.Function = Step;      break;
  }

  ierr = IGAOptionsAlias("-d",    "2", "-iga_dim");
  ierr = IGAOptionsAlias("-N",   "16", "-iga_elements");
  ierr = IGAOptionsAlias("-L", "-1,1", "-iga_limits");
  ierr = IGAOptionsAlias("-p",  NULL,  "-iga_degree");
  ierr = IGAOptionsAlias("-k",  NULL,  "-iga_continuity");
  ierr = IGAOptionsAlias("-q",  NULL,  "-iga_quadrature");

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDof(iga,1);CHKERRQ(ierr);
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  PetscInt dim;
  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);

  Mat A;
  Vec x,b;
  ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&b);CHKERRQ(ierr);
  ierr = IGASetFormSystem(iga,System,&app);CHKERRQ(ierr);
  ierr = IGAComputeSystem(iga,A,b);CHKERRQ(ierr);

  KSP ksp;
  ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  PetscInt dir;
  for (dir=0; dir<dim; dir++) {
    ierr = IGASetRuleType(iga,dir,IGA_RULE_LEGENDRE);CHKERRQ(ierr);
    ierr = IGASetRuleSize(iga,dir,10);CHKERRQ(ierr);
  }
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  PetscReal L2error;
  ierr = IGAComputeErrorNorm(iga,0,x,Exact,&L2error,&app);CHKERRQ(ierr);

  if (print_error) {ierr = PetscPrintf(PETSC_COMM_WORLD,"L2 error = %g\n",(double)L2error);CHKERRQ(ierr);}
  if (check_error) {if (L2error > 1e-3) SETERRQ1(PETSC_COMM_WORLD,1,"L2 error=%g\n",(double)L2error);}
  if (draw&&dim<3) {ierr = IGADrawVec(iga,x,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}

  if (save) {ierr = IGAWrite   (iga,  "L2Projection-geometry.dat");CHKERRQ(ierr);}
  if (save) {ierr = IGAWriteVec(iga,x,"L2Projection-solution.dat");CHKERRQ(ierr);}

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
