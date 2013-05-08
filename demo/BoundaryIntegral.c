/*
  This code solves the Laplace problem where the boundary conditions
  can be changed from Nuemann to Dirichlet via the commandline. While
  its primary use is in regression tests for PetIGA, it also
  demonstrates how boundary integrals may be performed to enforce
  things like Neumann conditions.

  keywords: steady, scalar, linear, testing, dimension independent,
  boundary integrals
 */
#include "petiga.h"

typedef struct {
  PetscInt dir;
  PetscInt side;
} AppCtx;

PETSC_STATIC_INLINE
PetscReal DOT(PetscInt dim,const PetscReal a[],const PetscReal b[])
{
  PetscInt i; PetscReal s = 0.0;
  for (i=0; i<dim; i++) s += a[i]*b[i];
  return s;
}

PETSC_STATIC_INLINE
PetscReal DEL2(PetscInt dim,const PetscReal a[dim][dim])
{
  PetscInt i; PetscReal s = 0.0;
  for (i=0; i<dim; i++) s += a[i][i];
  return s;
}

#undef  __FUNCT__
#define __FUNCT__ "System"
PetscErrorCode System(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  PetscInt dim = p->dim;
  PetscInt nen = p->nen;
  const PetscReal (*N1)[dim]; //*((PetscReal**)&N1) = p->shape[1];
  IGAPointGetShapeFuns(p,1,(const PetscReal **)&N1);
  PetscInt a,b;
  for (a=0; a<nen; a++) {
    for (b=0; b<nen; b++)
      K[a*nen+b] = DOT(dim,N1[a],N1[b]);
    F[a] = 0.0;
  }
  return 0;
}

#undef  __FUNCT__
#define __FUNCT__ "Neumann"
PetscErrorCode Neumann(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  PetscInt nen = p->nen;
  const PetscReal *N0 = p->shape[0];
  PetscInt a;
  for (a=0; a<nen; a++)
    F[a] = N0[a] * 1.0;
  return 0;
}

#undef  __FUNCT__
#define __FUNCT__ "SystemCollocation"
PetscErrorCode SystemCollocation(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  PetscInt nen = p->nen;
  PetscInt dim = p->dim;

  const PetscReal (*N2)[dim][dim]; //*((PetscReal**)&N2) = p->shape[2];
  IGAPointGetShapeFuns(p,2,(const PetscReal**)&N2);

  PetscInt a;
  for (a=0; a<nen; a++)
    K[a] = -DEL2(dim,N2[a]);
  F[0] = 0.0;

  return 0;
}

#undef  __FUNCT__
#define __FUNCT__ "DirichletCollocation"
PetscErrorCode DirichletCollocation(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  PetscInt nen = p->nen;
  const PetscReal *N0 = p->shape[0];
  PetscInt a;
  for (a=0; a<nen; a++)
    K[a] = N0[a];
  F[0] = 1.0;
  return 0;
}

#undef  __FUNCT__
#define __FUNCT__ "NeumannCollocation"
PetscErrorCode NeumannCollocation(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  PetscInt nen = p->nen;
  PetscInt dim = p->dim;
  const PetscReal (*N1)[dim];
  const PetscReal *normal = p->normal;
  IGAPointGetShapeFuns(p,1,(const PetscReal**)&N1);
  PetscInt a;
  for (a=0; a<nen; a++)
    K[a] = DOT(dim,N1[a],normal);
  F[0] = 1.0;
  return 0;
}
PetscErrorCode Neumann0Collocation(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{(void)NeumannCollocation(p,K,F,ctx); F[0] = 0.0; return 0;}

#undef  __FUNCT__
#define __FUNCT__ "Error"
PetscErrorCode Error(IGAPoint p,const PetscScalar *U,PetscInt n,PetscScalar *S,void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;
  PetscScalar u;
  IGAPointFormValue(p,U,&u);
  PetscReal x;
  if (user->side == 0)
    x = 1 - p->point[user->dir] + 1;
  else
    x = p->point[user->dir] + 1;
  PetscReal e = u - x;
  S[0] = e*e;
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[]) {

  PetscErrorCode  ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  AppCtx user;
  user.dir  = 0;
  user.side = 1;

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dir", "direction",__FILE__,user.dir, &user.dir, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-side","side",     __FILE__,user.side,&user.side,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDof(iga,1);CHKERRQ(ierr);
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  if (iga->dim < 1) {ierr = IGASetDim(iga,3);CHKERRQ(ierr);}

  PetscInt dim;
  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);

  IGABoundary bnd;
  PetscInt d =  !user.side;
  PetscInt n = !!user.side;
  if (!iga->collocation) {
    ierr = IGAGetBoundary(iga,user.dir,d,&bnd);CHKERRQ(ierr);
    ierr = IGABoundarySetValue(bnd,0,1.0);CHKERRQ(ierr);
    ierr = IGAGetBoundary(iga,user.dir,n,&bnd);CHKERRQ(ierr);
    ierr = IGABoundarySetUserSystem(bnd,Neumann,PETSC_NULL);CHKERRQ(ierr);
  } else {
    PetscInt dir,side;
    for (dir=0; dir<dim; dir++) {
      for (side=0; side<2; side++) {
        ierr = IGAGetBoundary(iga,dir,side,&bnd);CHKERRQ(ierr);
        ierr = IGABoundarySetUserSystem(bnd,Neumann0Collocation,PETSC_NULL);CHKERRQ(ierr);
      }
    }
    ierr = IGAGetBoundary(iga,user.dir,d,&bnd);CHKERRQ(ierr);
    ierr = IGABoundarySetUserSystem(bnd,DirichletCollocation,PETSC_NULL);CHKERRQ(ierr);
    ierr = IGAGetBoundary(iga,user.dir,n,&bnd);CHKERRQ(ierr);
    ierr = IGABoundarySetUserSystem(bnd,NeumannCollocation,PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  Mat A;
  Vec x,b;
  ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&b);CHKERRQ(ierr);
  if (!iga->collocation) {
    ierr = IGASetUserSystem(iga,System,PETSC_NULL);CHKERRQ(ierr);
  } else {
    ierr = IGASetUserSystem(iga,SystemCollocation,PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = IGAComputeSystem(iga,A,b);CHKERRQ(ierr);

  KSP ksp;
  ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  PetscScalar error = 0;
  ierr = IGAFormScalar(iga,x,1,&error,Error,&user);CHKERRQ(ierr);
  error = PetscSqrtReal(PetscRealPart(error));
  ierr = PetscPrintf(PETSC_COMM_WORLD,"L2 error = %G\n",error);CHKERRQ(ierr);

  PetscBool draw = PETSC_TRUE;
  if (draw && dim <= 2) {ierr = VecView(x,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
