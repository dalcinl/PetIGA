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

PetscReal Solution(PetscReal x, PetscReal y)
{
  PetscReal pi = M_PI;
  return sin(pi*x) * sin(pi*y);
}

PetscErrorCode Exact(IGAPoint p,PetscInt order,PetscScalar value[],void *ctx)
{
  PetscReal x = p->point[0];
  PetscReal y = p->point[1];
  value[0] = Solution(x,y);
  return 0;
}

int main(int argc, char *argv[]) {

  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  ierr = IGAOptionsAlias("-N",  "16", "-iga_elements");
  ierr = IGAOptionsAlias("-p", NULL,  "-iga_degree");
  ierr = IGAOptionsAlias("-k", NULL,  "-iga_continuity");
  ierr = IGAOptionsAlias("-q", NULL,  "-iga_quadrature");

  PetscLogDouble tic,toc;
  ierr = PetscTime(&tic);

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDim(iga,2);CHKERRQ(ierr);
  ierr = IGASetDof(iga,1);CHKERRQ(ierr);
  ierr = IGASetOrder(iga,1);CHKERRQ(ierr);
  PetscInt dir,side;
  for (dir=0; dir<2; dir++)
    for (side=0; side<2; side++)
      {ierr = IGASetBoundaryValue(iga,dir,side,0,0.0);CHKERRQ(ierr);}
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  Mat A;
  Vec x,b;
  ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&b);CHKERRQ(ierr);
  ierr = IGASetFormSystem(iga,System,NULL);CHKERRQ(ierr);

  PetscLogDouble ta1,ta2,ta;
  ierr = PetscTime(&ta1);
  ierr = IGAComputeSystem(iga,A,b);CHKERRQ(ierr);
  ierr = PetscTime(&ta2);
  ta = ta2-ta1;

  KSP ksp;
  ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
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

  for (dir=0; dir<2; dir++) {
    ierr = IGASetRuleType(iga,dir,IGA_RULE_LEGENDRE);CHKERRQ(ierr);
    ierr = IGASetRuleSize(iga,dir,10);CHKERRQ(ierr);
  }
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  PetscReal error = 0;
  ierr = IGAComputeErrorNorm(iga,0,x,Exact,&error,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,
                     "Error=%E , ||b-Ax||=%E in %D its , "
                     "Time: %f [assembly: %f (%.0f\%), solve: %f (%.0f\%)]\n",
                     error,rnorm,its,tt,ta,ta/tt*100,ts,ts/tt*100);CHKERRQ(ierr);

  PetscBool draw = IGAGetOptBool(NULL,"-draw",PETSC_FALSE);
  if (draw) {ierr = IGADrawVec(iga,x,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
