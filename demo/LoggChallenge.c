/*
This code was written in response to a challenge posed by Anders Logg
on Google+. The challenge:

   Solve the partial diferential equation

      -Laplacian(u) = f

   with homogeneous Dirichlet boundary conditions on the unit square for

      f(x,y) = 2 pi^2 sin(pi*x) * sin(pi*y).

   Who can compute a solution with an (L2) error smaller than 10^-6?
*/
#include "petiga.h"

PetscReal Forcing(PetscReal x, PetscReal y)
{
  PetscReal pi = M_PI;
  return 2*pi*pi * sin(pi*x) * sin(pi*y);
}

#undef  __FUNCT__
#define __FUNCT__ "System"
PetscErrorCode System(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  PetscReal x = p->point[0];
  PetscReal y = p->point[1];
  PetscReal f = Forcing(x,y);
  PetscReal *N0 = p->shape[0];
  PetscReal (*N1)[2] = (PetscReal(*)[2]) p->shape[1];
  PetscInt a,b,nen=p->nen;
  for (a=0; a<nen; a++) {
    PetscReal Na   = N0[a];
    PetscReal Na_x = N1[a][0];
    PetscReal Na_y = N1[a][1];
    for (b=0; b<nen; b++) {
      PetscReal Nb_x = N1[b][0];
      PetscReal Nb_y = N1[b][1];
      K[a*nen+b] = Na_x*Nb_x + Na_y*Nb_y;
    }
    F[a] = Na * f;
  }
  return 0;
}

PetscReal Exact(PetscReal x, PetscReal y)
{
  PetscReal pi = M_PI;
  return sin(pi*x) * sin(pi*y);
}

#undef  __FUNCT__
#define __FUNCT__ "Error"
PetscErrorCode Error(IGAPoint p,const PetscScalar *U,PetscInt n,PetscScalar *S,void *ctx)
{
  PetscReal x = p->point[0];
  PetscReal y = p->point[1];
  PetscReal u_exact = Exact(x,y);
  PetscScalar u;
  IGAPointFormValue(p,U,&u);
  PetscReal e = PetscAbsScalar(u - u_exact);
  S[0] = e*e;
  return 0;
}

#if PETSC_VERSION_LT(3,4,0)
#undef  PetscTime
#define PetscTime PetscGetTime
#endif

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[]) {

  PetscErrorCode  ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  PetscLogDouble tic,toc;
  ierr = PetscTime(&tic);

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDim(iga,2);CHKERRQ(ierr);
  ierr = IGASetDof(iga,1);CHKERRQ(ierr);

  IGABoundary bnd;
  PetscInt dir,side;
  for (dir=0; dir<2; dir++) {
    for (side=0; side<2; side++) {
      ierr = IGAGetBoundary(iga,dir,side,&bnd);CHKERRQ(ierr);
      ierr = IGABoundarySetValue(bnd,0,0.0);CHKERRQ(ierr);
    }
  }

  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  Mat A;
  Vec x,b;
  ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&b);CHKERRQ(ierr);
  ierr = IGASetUserSystem(iga,System,NULL);CHKERRQ(ierr);

  PetscLogDouble ta1,ta2,ta;
  ierr = PetscTime(&ta1);
  ierr = IGAComputeSystem(iga,A,b);CHKERRQ(ierr);
  ierr = PetscTime(&ta2);
  ta = ta2-ta1;

  KSP ksp;
  ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  PetscLogDouble ts1,ts2,ts;
  ierr = PetscTime(&ts1);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  ierr = PetscTime(&ts2);
  ts = ts2-ts1;

  ierr = PetscTime(&toc);
  PetscLogDouble tt = toc-tic;

  Vec r;
  PetscReal rnorm;
  ierr = VecDuplicate(b,&r);CHKERRQ(ierr);
  ierr = MatMult(A,x,r);CHKERRQ(ierr);
  ierr = VecAYPX(r,-1.0,b);CHKERRQ(ierr);
  ierr = VecNorm(r,NORM_2,&rnorm);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);

  PetscInt its;
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);

  ierr = IGAReset(iga);CHKERRQ(ierr);
  for (dir=0; dir<2; dir++) {
    IGARule rule;
    ierr = IGAGetRule(iga,dir,&rule);CHKERRQ(ierr);
    ierr = IGARuleInit(rule,9);CHKERRQ(ierr);
  }
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  PetscScalar error = 0;
  ierr = IGAFormScalar(iga,x,1,&error,Error,NULL);CHKERRQ(ierr);
  error = PetscSqrtReal(PetscRealPart(error));

  ierr = PetscPrintf(PETSC_COMM_WORLD,
                     "Error=%E , ||b-Ax||=%E in %D its , "
                     "Time: %f [assembly: %f (%.0f\%), solve: %f (%.0f\%)]\n",
                     error,rnorm,its,tt,ta,ta/tt*100,ts,ts/tt*100);CHKERRQ(ierr);

  PetscBool draw = PETSC_FALSE;
  ierr = PetscOptionsGetBool(0,"-draw",&draw,0);CHKERRQ(ierr);
  if (draw) {
    ierr = VecView(x,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  }

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
