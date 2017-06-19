#include "petiga.h"

static void Function(PetscReal xyz[3],PetscReal *F)
{
  PetscReal x = xyz[0];
  PetscReal y = xyz[1];
  PetscReal z = xyz[2];
  *F  = 0;
  *F += x*x*x;
  *F += y*y*y;
  *F += z*z*z;
  *F += x*x + y*y + z*z;
  *F += x*y + x*z + y*z;
  *F += 1;
}
static void Gradient(PetscReal xyz[3],PetscReal G[3])
{
  PetscReal x = xyz[0];
  PetscReal y = xyz[1];
  PetscReal z = xyz[2];
  G[0] = 3*x*x + 2*x + (y+z);
  G[1] = 3*y*y + 2*y + (x+z);
  G[2] = 3*z*z + 2*z + (x+y);
}
static void Hessian(PetscReal xyz[3],PetscReal H[3][3])
{
  PetscReal x = xyz[0];
  PetscReal y = xyz[1];
  PetscReal z = xyz[2];
  H[0][0] = 6*x+2; H[0][1] = 1;     H[0][2] = 1;
  H[1][0] = 1;     H[1][1] = 6*y+2; H[1][2] = 1;
  H[2][0] = 1;     H[2][1] = 1;     H[2][2] = 6*z+2;
}
static void ThirdDer(PetscReal xyz[3],PetscReal D[3][3][3])
{
  PetscInt i,j,k;
  for (i=0; i<3; i++)
    for (j=0; j<3; j++)
      for (k=0; k<3; k++)
        D[i][j][k] = 0;
  D[0][0][0] = 6;
  D[1][1][1] = 6;
  D[2][2][2] = 6;
}

PetscErrorCode System(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  PetscInt nen = p->nen;

  PetscReal x[3] = {0,0,0};
  IGAPointFormGeomMap(p,x);
  PetscReal f;
  Function(x,&f);

  const PetscReal *N = (typeof(N)) p->shape[0];

  PetscInt a,b;
  for (a=0; a<nen; a++) {
    for (b=0; b<nen; b++)
      K[a*nen+b] = N[a] * N[b];
    F[a] = N[a] * f;
  }
  return 0;
}

#define CheckCLOSE(rtol,atol,a,b)                   \
  do {                                              \
    double _a = (a), _b = (b);                      \
    if (fabs(_a-_b) > atol+rtol*fabs(_b))           \
      SETERRQ2(PETSC_COMM_SELF,1,"%f != %f",_a,_b); \
  } while(0)

PetscErrorCode Test(IGAProbe prb,PetscReal u[])
{
  PetscInt       i,j,k,dim = prb->dim;
  PetscScalar    a0,a1[dim],a2[dim][dim],a3[dim][dim][dim];
  PetscReal      X[3] = {0.0, 0.0, 0.0};
  PetscReal      b0,b1[3],b2[3][3],b3[3][3][3];
  PetscInt       p_min = prb->p[0];
  PetscErrorCode ierr;
  PetscFunctionBegin;

  for (i=0; i<dim; i++) p_min = PetscMin(p_min,prb->p[i]);

  ierr = IGAProbeSetPoint(prb,u);CHKERRQ(ierr);

  ierr = IGAProbeFormValue(prb,&a0);CHKERRQ(ierr);
  ierr = IGAProbeFormGrad (prb,&a1[0]);CHKERRQ(ierr);
  ierr = IGAProbeFormHess (prb,&a2[0][0]);CHKERRQ(ierr);
  ierr = IGAProbeFormDer3 (prb,&a3[0][0][0]);CHKERRQ(ierr);

  ierr = IGAProbeGeomMap(prb,X);CHKERRQ(ierr);

  Function(X,&b0);
  CheckCLOSE(1e-6,1e-4,PetscRealPart(a0), b0);

  Gradient(X,b1);
  for (i=0; i<dim; i++)
    CheckCLOSE(1e-5,1e-3,PetscRealPart(a1[i]), b1[i]);

  if (p_min < 2) PetscFunctionReturn(0);
  Hessian(X,b2);
  for (i=0; i<dim; i++)
    for (j=0; j<dim; j++)
      CheckCLOSE(1e-4,1e-2,PetscRealPart(a2[i][j]), b2[i][j]);

  if (p_min < 3) PetscFunctionReturn(0);
  ThirdDer(X,b3);
  for (i=0; i<dim; i++)
    for (j=0; j<dim; j++)
      for (k=0; k<dim; k++)
        CheckCLOSE(1e-3,1e-1,PetscRealPart(a3[i][j][k]), b3[i][j][k]);

  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  PetscBool      collective = PETSC_TRUE;
  IGA            iga;
  Vec            vec;
  IGAProbe       prb;
  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);
  /*
  ierr = IGAOptionsAlias("-dim",  "2", "-iga_dim");CHKERRQ(ierr);
  ierr = IGAOptionsAlias("-N",   "16", "-iga_elements");CHKERRQ(ierr);
  ierr = IGAOptionsAlias("-p",    "3", "-iga_degree");CHKERRQ(ierr);
  ierr = IGAOptionsAlias("-C",   "-1", "-iga_continuity");CHKERRQ(ierr);
  ierr = IGAOptionsAlias("-L",  "0,1", "-iga_limits");CHKERRQ(ierr);
  */
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","IGAProbe Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-collective","Collective evaluation",__FILE__,collective,&collective,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDof(iga,1);CHKERRQ(ierr);
  {
    PetscInt i; IGAAxis axis;
    for (i=0; i<3; i++) {
      ierr = IGAGetAxis(iga,i,&axis);CHKERRQ(ierr);
      ierr = IGAAxisSetDegree(axis,3);CHKERRQ(ierr);
    }
  }
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);

  if (iga->dim < 1) {ierr = IGASetDim(iga,2);CHKERRQ(ierr);}
  ierr = IGASetOrder(iga,4);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  ierr = IGACreateVec(iga,&vec);CHKERRQ(ierr);

  {
    Mat A;
    Vec b;
    KSP ksp;
    ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
    ierr = IGACreateVec(iga,&b);CHKERRQ(ierr);
    ierr = IGASetFormSystem(iga,System,NULL);CHKERRQ(ierr);
    ierr = IGAComputeSystem(iga,A,b);CHKERRQ(ierr);

    ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
    ierr = KSPSetType(ksp,KSPCG);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp,0.0,100*PETSC_MACHINE_EPSILON,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
    ierr = KSPSolve(ksp,b,vec);CHKERRQ(ierr);

    ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = VecDestroy(&b);CHKERRQ(ierr);
  }

  ierr = IGAProbeCreate(iga,vec,&prb);CHKERRQ(ierr);
  ierr = IGAProbeSetOrder(prb,3);CHKERRQ(ierr);
  ierr = IGAProbeSetCollective(prb,collective);CHKERRQ(ierr);

  {
    MPI_Comm comm;PetscMPIInt size,rank;
    ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
    if (collective || rank==0) {
      PetscReal u[3] = {0.0, 0.0, 0.0};
      ierr = Test(prb,u);CHKERRQ(ierr);
    }
    if (collective) {
      PetscReal u[3] = {0.5, 0.5, 0.5};
      ierr = Test(prb,u);CHKERRQ(ierr);
    }
    if (collective || rank==size-1) {
      PetscReal u[3] = {1.0, 1.0, 1.0};
      ierr = Test(prb,u);CHKERRQ(ierr);
    }
  }

  ierr = IGAProbeDestroy(&prb);CHKERRQ(ierr);
  ierr = VecDestroy(&vec);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
