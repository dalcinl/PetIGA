#include "petiga.h"

#undef  __FUNCT__
#define __FUNCT__ "System"
PetscErrorCode System(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  PetscInt nen,dim;
  IGAPointGetSizes(p,0,&nen,0);
  IGAPointGetDims(p,&dim,0);

  const PetscReal *N1;
  IGAPointGetShapeFuns(p,1,&N1);

  PetscInt a,b,i;
  for (a=0; a<nen; a++) {
    for (b=0; b<nen; b++) {
      PetscScalar Kab = 0.0;
      for (i=0; i<dim; i++)
        Kab += N1[a*dim+i]*N1[b*dim+i];
      K[a*nen+b] = Kab;
    }
  }
  return 0;
}



#undef  __FUNCT__
#define __FUNCT__ "Neumann"
PetscErrorCode Neumann(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  PetscReal *N0 = p->shape[0];
  PetscInt a,nen=p->nen;
  for (a=0; a<nen; a++) {
    PetscReal Na   = N0[a];
    F[a] = Na * 1.0;
  }
  return 0;
}

typedef struct { 
  PetscInt dir;
  PetscInt side;
} AppCtx;

#undef  __FUNCT__
#define __FUNCT__ "Error"
PetscErrorCode Error(IGAPoint p,const PetscScalar *U,PetscInt n,PetscScalar *S,void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;
  PetscScalar u;
  IGAPointFormValue(p,U,&u);
  PetscReal x;
  if (user->side == 0)
    x = 1 - p->point[user->dir];
  else
    x = p->point[user->dir];
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

  IGABoundary bnd;
  PetscInt d = !user.side; 
  PetscInt n = !!user.side;
  ierr = IGAGetBoundary(iga,user.dir,d,&bnd);CHKERRQ(ierr);
  ierr = IGABoundarySetValue(bnd,0,0.0);CHKERRQ(ierr);
  ierr = IGAGetBoundary(iga,user.dir,n,&bnd);CHKERRQ(ierr); 
  ierr = IGABoundarySetUserSystem(bnd,Neumann,PETSC_NULL);CHKERRQ(ierr);

  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  Mat A;
  Vec x,b;
  ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&b);CHKERRQ(ierr);
  ierr = IGASetUserSystem(iga,System,PETSC_NULL);CHKERRQ(ierr);
  ierr = IGAComputeSystem(iga,A,b);CHKERRQ(ierr);
  
  KSP ksp;
  ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  PetscInt dim;
  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
  if (dim <= 2) {ierr = VecView(x,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}

  PetscScalar error = 0;
  ierr = IGAFormScalar(iga,x,1,&error,Error,&user);CHKERRQ(ierr);
  error = PetscSqrtReal(PetscRealPart(error));
  ierr = PetscPrintf(PETSC_COMM_WORLD,"L2 error = %G\n",error);CHKERRQ(ierr);
  
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  PetscBool flag = PETSC_FALSE;
  PetscReal secs = -1;
  ierr = PetscOptionsHasName(0,"-sleep",&flag);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(0,"-sleep",&secs,0);CHKERRQ(ierr);
  if (flag) {ierr = PetscSleep(secs);CHKERRQ(ierr);}

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
