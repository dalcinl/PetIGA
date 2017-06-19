#include "petiga.h"

#define DO(i) for (i=0; i<dim; i++)

PETSC_STATIC_INLINE PetscReal Sum1(PetscInt dim,const PetscReal x[])
{
  PetscInt i; PetscReal r = 0;
  DO(i) r += x[i];
  return r;
}

PETSC_STATIC_INLINE PetscReal Sum2(PetscInt dim,const PetscReal x[])
{
  PetscInt i; PetscReal r = 0;
  DO(i) r += x[i]*x[i];
  return r;
}

PETSC_STATIC_INLINE PetscReal Prod(PetscInt dim,const PetscReal x[])
{
  PetscInt i; PetscReal r = 1;
  DO(i) r *= x[i];
  return r;
}

PetscErrorCode Exact(IGAPoint p,PetscInt order,PetscScalar value[],void *ctx)
{
  PetscInt  i,j,dim = p->dim;
  PetscReal x[3];
  IGAPointFormGeomMap(p,x);
  switch (order) {
  case 0:
    value[0] = 1;
    value[1] = Sum1(dim,x);
    value[2] = Sum2(dim,x);
    value[3] = Prod(dim,x);
    break;
  case 1:
    DO(i) value[0*dim+i] = 0;
    DO(i) value[1*dim+i] = 1;
    DO(i) value[2*dim+i] = 2*x[i];
    DO(i) value[3*dim+i] = Prod(dim,x)/x[i];
    break;
  case 2:
    DO(i) DO(j) value[0*dim*dim+i*dim+j] = 0;
    DO(i) DO(j) value[1*dim*dim+i*dim+j] = 0;
    DO(i) DO(j) value[2*dim*dim+i*dim+j] = (i==j) ? 2 : 0;
    DO(i) DO(j) value[3*dim*dim+i*dim+j] = (i==j) ? 0 : (Prod(dim,x)/(x[i]*x[j]));
    break;
  }
  return 0;
}

PetscErrorCode System(IGAPoint p,PetscScalar *KK,PetscScalar *FF,void *ctx)
{
  PetscInt dof = p->dof;
  PetscInt nen = p->nen;
  const PetscReal *N = (typeof(N)) p->shape[0];
  PetscScalar (*K)[dof][nen][dof] = (typeof(K)) KK;
  PetscScalar (*F)[dof] = (typeof(F)) FF;

  PetscScalar u[dof];
  (void)Exact(p,0,u,NULL);

  PetscInt a,b,i;
  for (a=0; a<nen; a++)
    for (b=0; b<nen; b++)
      for (i=0; i<dof; i++)
        K[a][i][b][i] = N[a] * N[b];
  for (a=0; a<nen; a++)
    for (i=0; i<dof; i++)
      F[a][i] = N[a] * u[i];

  return 0;
}

#define AssertEQUAL(a,b)                              \
  do {                                                \
    const PetscReal tol = PETSC_SQRT_MACHINE_EPSILON; \
    if (PetscAbsReal((a)-(b)) > tol)                  \
      SETERRQ2(PETSC_COMM_SELF,1,"%.16f != %.16f",    \
               (double)(a),(double)(b));              \
  } while(0)

int main(int argc, char *argv[])
{
  PetscInt       i,dim;
  IGA            iga;
  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  ierr = IGAOptionsAlias("-d",  "2", "-iga_dim");
  ierr = IGAOptionsAlias("-n",  "8", "-iga_elements");
  ierr = IGAOptionsAlias("-p", NULL, "-iga_degree");
  ierr = IGAOptionsAlias("-k", NULL, "-iga_continuity");
  ierr = IGAOptionsAlias("-q", NULL, "-iga_rule_size");
  ierr = IGAOptionsAlias("-r", NULL, "-iga_rule_type");

  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDof(iga,4);CHKERRQ(ierr);
  for (i=0; i<3; i++) {ierr = IGASetRuleType(iga,i,IGA_RULE_LEGENDRE);CHKERRQ(ierr);}
  for (i=0; i<3; i++) {ierr = IGASetRuleSize(iga,i,3);CHKERRQ(ierr);}
  ierr = IGASetOrder(iga,2);CHKERRQ(ierr);
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);
  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);

  PetscReal normL2[4],seminormH1[4],seminormH2[4];
  ierr = IGAComputeErrorNorm(iga,0,NULL,Exact,normL2,NULL);CHKERRQ(ierr);
  ierr = IGAComputeErrorNorm(iga,1,NULL,Exact,seminormH1,NULL);CHKERRQ(ierr);
  ierr = IGAComputeErrorNorm(iga,2,NULL,Exact,seminormH2,NULL);CHKERRQ(ierr);

# define SQRT PetscSqrtReal
  /*                            function:  1 - sum(x[i])       - sum(x[i]^2)       - prod(x[i])     */
  PetscReal expectedL2[3][4] = {/*dim=1*/{ 1 , 1/SQRT(3)       , 1/SQRT(5)         , 1/SQRT(3)       },
                                /*dim=2*/{ 1 , SQRT(7)/SQRT(6) , SQRT(28)/SQRT(45) , 1/SQRT(9)       },
                                /*dim=3*/{ 1 , SQRT(5)/SQRT(2) , SQRT(19)/SQRT(15) , 1/SQRT(27)      }};
  PetscReal expectedH1[3][4] = {/*dim=1*/{ 0 , 1               , 2/SQRT(3)         , 1               },
                                /*dim=2*/{ 0 , SQRT(2)         , SQRT(8)/SQRT(3)   , SQRT(2)/SQRT(3) },
                                /*dim=3*/{ 0 , SQRT(3)         , 2                 , 1/SQRT(3)       }};
  PetscReal expectedH2[3][4] = {/*dim=1*/{ 0 , 0               , 2                 , 0               },
                                /*dim=2*/{ 0 , 0               , SQRT(8)           , SQRT(2)         },
                                /*dim=3*/{ 0 , 0               , SQRT(12)          , SQRT(2)         }};
  for (i=0; i<4; i++) AssertEQUAL(expectedL2[dim-1][i],normL2[i]);
  for (i=0; i<4; i++) AssertEQUAL(expectedH1[dim-1][i],seminormH1[i]);
  for (i=0; i<4; i++) AssertEQUAL(expectedH2[dim-1][i],seminormH2[i]);


  Mat A; Vec x,b; KSP ksp;
  ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&b);CHKERRQ(ierr);
  ierr = IGASetFormSystem(iga,System,NULL);CHKERRQ(ierr);
  ierr = IGAComputeSystem(iga,A,b);CHKERRQ(ierr);
  ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  ierr = IGAComputeErrorNorm(iga,0,x,Exact,normL2,NULL);CHKERRQ(ierr);
  ierr = IGAComputeErrorNorm(iga,1,x,Exact,seminormH1,NULL);CHKERRQ(ierr);
  ierr = IGAComputeErrorNorm(iga,2,x,Exact,seminormH2,NULL);CHKERRQ(ierr);
  for (i=0; i<4; i++) AssertEQUAL(0,normL2[i]);
  for (i=0; i<4; i++) AssertEQUAL(0,seminormH1[i]);
#if !defined(PETSC_USE_REAL_SINGLE)
  for (i=0; i<4; i++) AssertEQUAL(0,seminormH2[i]);
#endif

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
