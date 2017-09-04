/*
  This code computes the L2 projection of a function defined in
  Function to a B-spline space. The space is adaptively refined by a
  brute force strategy: which knot do we insert such that the global
  L2 error is reduced the most?

  keywords: scalar, linear
 */
#include "petiga.h"

#define SQ(A) (A)*(A)

PetscScalar Function(PetscReal x, PetscReal y, PetscReal z)
{
  PetscScalar x0=0.85,y0=0.45;
  PetscScalar sigx=0.125,sigy=0.2;
  return exp( -( SQ(x-x0) / (2.*SQ(sigx)) + SQ(y-y0) / (2.*SQ(sigy) ) ) );
}

PetscErrorCode System(IGAPoint p,PetscScalar *K,PetscScalar *F,void *ctx)
{
  PetscInt nen = p->nen;
  
  PetscReal x[3] = {0,0,0};
  IGAPointFormGeomMap(p,x);
  PetscScalar f = Function(x[0],x[1],x[2]);
  
  const PetscReal *N;
  IGAPointGetShapeFuns(p,0,(const PetscReal**)&N);
  
  PetscInt a,b;
  for (a=0; a<nen; a++) {
    for (b=0; b<nen; b++) {
      K[a*nen+b] = N[a] * N[b];
    }
    F[a] = N[a] * f;
  }
  return 0;
}

PetscErrorCode Error(IGAPoint p,const PetscScalar *U,PetscInt n,PetscScalar *S,void *ctx)
{
  PetscReal x[3] = {0,0,0};
  IGAPointFormGeomMap(p,x);
  PetscScalar f = Function(x[0],x[1],x[2]);
  
  PetscScalar u;
  IGAPointFormValue(p,U,&u);
  
  PetscReal e = PetscAbsScalar(u - f);
  S[0] = e*e;
  S[1] = f;

  return 0;
}

#define MAX_BREAKS 1000

PetscErrorCode ComputeError(PetscInt dim,PetscInt p,PetscInt C,PetscReal (*U)[MAX_BREAKS],PetscInt *N,PetscReal *error)
{
  PetscErrorCode  ierr;
  PetscInt i;
  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDim(iga,dim);CHKERRQ(ierr);
  ierr = IGASetDof(iga,1);CHKERRQ(ierr);
  for (i=0; i<dim; i++) {
    IGAAxis axis;
    ierr = IGAGetAxis(iga,i,&axis);CHKERRQ(ierr);
    ierr = IGAAxisSetDegree(axis,p);CHKERRQ(ierr);
    ierr = IGAAxisInitBreaks(axis,N[i],U[i],C);CHKERRQ(ierr);
  }
  ierr = IGASetUp(iga);CHKERRQ(ierr);
  
  Mat A;
  Vec x,b;
  ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&b);CHKERRQ(ierr);
  ierr = IGASetFormSystem(iga,System,NULL);CHKERRQ(ierr);
  ierr = IGAComputeSystem(iga,A,b);CHKERRQ(ierr);

  KSP ksp;
  ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  PetscScalar scalar[2];
  ierr  = IGAComputeScalar(iga,x,2,&scalar[0],Error,NULL);CHKERRQ(ierr);
  PetscReal errors[2];
  errors[0] = PetscSqrtReal(PetscRealPart(scalar[0]));
  errors[1] = PetscSqrtReal(PetscRealPart(scalar[1]));
  *error = errors[0]/errors[1];

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);
  return 0;
}

PetscErrorCode FindBestKnotToInsert(PetscInt dim,PetscInt idim,PetscInt p,PetscInt C,PetscReal (*U)[MAX_BREAKS],
				    PetscInt *N,PetscInt *insert,PetscReal *min_error) 
{
  PetscErrorCode  ierr;
  PetscInt i,j;

  // Initialize other dimensions to values in U
  PetscReal Ur[3][MAX_BREAKS];
  PetscInt  Nr[3];
  for(i=0;i<dim;i++) {
    if(i != idim) for(j=0;j<N[i];j++) Ur[i][j] = U[i][j];
    Nr[i] = N[i];
  }
  Nr[idim] += 1;

  *insert    = -1;
  *min_error = 1e20;

  // Try refining each nonzero span
  for(i=0;i<N[idim]-1;i++){  
    for(j=0;j<N[idim]+1;j++){
      if(j<=i)   Ur[idim][j] = U[idim][j]; 
      if(j==i+1) Ur[idim][j] = (U[idim][j-1]+U[idim][j])/2;
      if(j>i+1)  Ur[idim][j] = U[idim][j-1]; 
    }
    PetscReal error;
    ierr = ComputeError(dim,p,C,&Ur[0],&Nr[0],&error);CHKERRQ(ierr);
    if(error < *min_error){
      *min_error = error;
      *insert    = i;
    }
  }

  return 0;
}

int main(int argc, char *argv[]) {

  PetscErrorCode  ierr;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  PetscReal U[3][MAX_BREAKS];
  PetscInt  N[3];

  PetscInt i,j;
  PetscInt dim = 2, p = 2, C = 1;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","Options","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim","dimension",__FILE__,dim,&dim,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-p","polynomial order",       __FILE__,p,&p,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-C","global continuity order",__FILE__,C,&C,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  // initial breaks (2x2 mesh)
  for (i=0; i<dim; i++) { U[i][0] = 0.; U[i][1] = .5; U[i][2] = 1.; N[i] = 3; }

  PetscInt step = 0;
  PetscReal min_error = 1.e20;
  while(step<50 && min_error > 1e-9){

    PetscInt insert[3],imin=-1;
    PetscReal error[3];
    min_error = 1.e20;

    for(i=0;i<dim;i++){
      ierr = FindBestKnotToInsert(dim,i,p,C,&U[0],&N[0],&insert[i],&error[i]);CHKERRQ(ierr);
      if(error[i] < min_error){
	min_error = error[i];
	imin = i;
      }
    }

    if(imin < 0) {
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"No minimum found.");
    }

    PetscReal in = 0.5*(U[imin][insert[imin]]+U[imin][insert[imin]+1]);
    for(j=N[imin];j>insert[imin];j--) U[imin][j] = U[imin][j-1];
    U[imin][insert[imin]+1] = in;
    N[imin] += 1;

    printf("Step %d: Error %.16e\n",(int)step,(double)min_error);

    step += 1;
  }
  
  
  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDim(iga,dim);CHKERRQ(ierr);
  ierr = IGASetDof(iga,1);CHKERRQ(ierr);
  for (i=0; i<dim; i++) {
    IGAAxis axis;
    ierr = IGAGetAxis(iga,i,&axis);CHKERRQ(ierr);
    ierr = IGAAxisSetDegree(axis,p);CHKERRQ(ierr);
    ierr = IGAAxisInitBreaks(axis,N[i],U[i],C);CHKERRQ(ierr);
  }
  ierr = IGASetUp(iga);CHKERRQ(ierr);
  
  Mat A;
  Vec x,b;
  ierr = IGACreateMat(iga,&A);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&x);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&b);CHKERRQ(ierr);
  ierr = IGASetFormSystem(iga,System,NULL);CHKERRQ(ierr);
  ierr = IGAComputeSystem(iga,A,b);CHKERRQ(ierr);

  KSP ksp;
  ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  
  ierr = VecView(x,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);

  ierr = IGAWrite(iga,"iga.dat");CHKERRQ(ierr);
  ierr = IGAWriteVec(iga,x,"sol.dat");CHKERRQ(ierr);

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = IGADestroy(&iga);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
