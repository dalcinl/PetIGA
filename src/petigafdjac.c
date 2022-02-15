#include "petiga.h"

static inline
PetscReal DeltaWP(PetscInt N,const PetscScalar U[])
{
  const PetscReal sqrteps = PETSC_SQRT_MACHINE_EPSILON;
  PetscInt i; PetscReal normU = 0;
  for (i=0; i<N; i++) normU += PetscRealPart(PetscConj(U[i])*U[i]);
  normU = PetscSqrtReal(normU);
  return sqrteps * PetscSqrtReal(1 + normU);
}

PetscErrorCode IGAFormJacobianFD(IGAPoint p,
                                 const PetscScalar U[],
                                 PetscScalar J[],PETSC_UNUSED void *ctx)
{
  typedef        IGAFormFunction IGAFormFunType;
  IGAForm        form = p->parent->parent->form;
  IGAFormFunType Function = form->ops->Function;
  void           *FunCtx  = form->ops->FunCtx;
  PetscInt       i,M = p->neq*p->dof;
  PetscInt       j,N = p->nen*p->dof;
  PetscScalar    *F,*G;
  PetscReal      h = DeltaWP(N,U);
  PetscErrorCode ierr;
  PetscScalar    *X = (PetscScalar*)U;

  PetscFunctionBegin;
  ierr = IGAPointGetWorkVec(p,&F);CHKERRQ(ierr);
  ierr = IGAPointGetWorkVec(p,&G);CHKERRQ(ierr);
  ierr = Function(p,U,F,FunCtx);CHKERRQ(ierr);
  for (j=0; j<N; j++) {
    PetscScalar Uj = X[j];
    X[j] = Uj + h;
    ierr = Function(p,X,G,FunCtx);CHKERRQ(ierr);
    for (i=0; i<M; i++) J[i*N+j] = (G[i]-F[i])/h;
    X[j] = Uj;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IGAFormIJacobianFD(IGAPoint p,
                                  PetscReal s,const PetscScalar V[],
                                  PetscReal t,const PetscScalar U[],
                                  PetscScalar J[],PETSC_UNUSED void *ctx)
{
  typedef        IGAFormIFunction IGAFormFunType;
  IGAForm        form = p->parent->parent->form;
  IGAFormFunType Function = form->ops->IFunction;
  void           *FunCtx  = form->ops->IFunCtx;
  PetscInt       i,M = p->neq*p->dof;
  PetscInt       j,N = p->nen*p->dof;
  PetscScalar    *F,*G;
  PetscReal      h = DeltaWP(N,U);
  PetscErrorCode ierr;
  PetscScalar    *Y = (PetscScalar*)V;
  PetscScalar    *X = (PetscScalar*)U;

  PetscFunctionBegin;
  ierr = IGAPointGetWorkVec(p,&F);CHKERRQ(ierr);
  ierr = IGAPointGetWorkVec(p,&G);CHKERRQ(ierr);
  ierr = Function(p,s,V,t,U,F,FunCtx);CHKERRQ(ierr);
  for (j=0; j<N; j++) {
    PetscScalar Vj = Y[j];
    PetscScalar Uj = X[j];
    Y[j] = Vj + h*s;
    X[j] = Uj + h;
    ierr = Function(p,s,Y,t,X,G,FunCtx);CHKERRQ(ierr);
    for (i=0; i<M; i++) J[i*N+j] = (G[i]-F[i])/h;
    Y[j] = Vj;
    X[j] = Uj;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IGAFormIEJacobianFD(IGAPoint p,
                                   PetscReal s, const PetscScalar V[],
                                   PetscReal t, const PetscScalar U[],
                                   PetscReal t0,const PetscScalar U0[],
                                   PetscScalar J[],PETSC_UNUSED void *ctx)
{
  typedef        IGAFormIEFunction IGAFormFunType;
  IGAForm        form = p->parent->parent->form;
  IGAFormFunType Function = form->ops->IEFunction;
  void           *FunCtx  = form->ops->IEFunCtx;
  PetscInt       i,M = p->neq*p->dof;
  PetscInt       j,N = p->nen*p->dof;
  PetscScalar    *F,*G;
  PetscReal      h = DeltaWP(N,U);
  PetscErrorCode ierr;
  PetscScalar    *Y = (PetscScalar*)V;
  PetscScalar    *X = (PetscScalar*)U;

  PetscFunctionBegin;
  ierr = IGAPointGetWorkVec(p,&F);CHKERRQ(ierr);
  ierr = IGAPointGetWorkVec(p,&G);CHKERRQ(ierr);
  ierr = Function(p,s,V,t,U,t0,U0,F,FunCtx);CHKERRQ(ierr);
  for (j=0; j<N; j++) {
    PetscScalar Vj = Y[j];
    PetscScalar Uj = X[j];
    Y[j] = Vj + h*s;
    X[j] = Uj + h;
    ierr = Function(p,s,Y,t,X,t0,U0,G,FunCtx);CHKERRQ(ierr);
    for (i=0; i<M; i++) J[i*N+j] = (G[i]-F[i])/h;
    Y[j] = Vj;
    X[j] = Uj;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IGAFormI2JacobianFD(IGAPoint p,
                                   PetscReal a,const PetscScalar A[],
                                   PetscReal v,const PetscScalar V[],
                                   PetscReal t,const PetscScalar U[],
                                   PetscScalar J[],PETSC_UNUSED void *ctx)
{
  typedef        IGAFormI2Function IGAFormFunType;
  IGAForm        form = p->parent->parent->form;
  IGAFormFunType Function = form->ops->I2Function;
  void           *FunCtx  = form->ops->IFunCtx;
  PetscInt       i,M = p->neq*p->dof;
  PetscInt       j,N = p->nen*p->dof;
  PetscScalar    *F,*G;
  PetscReal      h = DeltaWP(N,U);
  PetscErrorCode ierr;
  PetscScalar    *Z = (PetscScalar*)A;
  PetscScalar    *Y = (PetscScalar*)V;
  PetscScalar    *X = (PetscScalar*)U;

  PetscFunctionBegin;
  ierr = IGAPointGetWorkVec(p,&F);CHKERRQ(ierr);
  ierr = IGAPointGetWorkVec(p,&G);CHKERRQ(ierr);
  ierr = Function(p,a,A,v,V,t,U,F,FunCtx);CHKERRQ(ierr);
  for (j=0; j<N; j++) {
    PetscScalar Aj = Z[j];
    PetscScalar Vj = Y[j];
    PetscScalar Uj = X[j];
    Z[j] = Aj + h*a;
    Y[j] = Vj + h*v;
    X[j] = Uj + h;
    ierr = Function(p,a,Z,v,Y,t,X,G,FunCtx);CHKERRQ(ierr);
    for (i=0; i<M; i++) J[i*N+j] = (G[i]-F[i])/h;
    Z[j] = Aj;
    Y[j] = Vj;
    X[j] = Uj;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IGAFormRHSJacobianFD(IGAPoint p,
                                    PetscReal t,const PetscScalar U[],
                                    PetscScalar J[],PETSC_UNUSED void *ctx)
{
  typedef        IGAFormRHSFunction IGAFormFunType;
  IGAForm        form = p->parent->parent->form;
  IGAFormFunType Function = form->ops->RHSFunction;
  void           *FunCtx  = form->ops->RHSFunCtx;
  PetscInt       i,M = p->neq*p->dof;
  PetscInt       j,N = p->nen*p->dof;
  PetscScalar    *F,*G;
  PetscReal      h = DeltaWP(N,U);
  PetscErrorCode ierr;
  PetscScalar    *X = (PetscScalar*)U;

  PetscFunctionBegin;
  ierr = IGAPointGetWorkVec(p,&F);CHKERRQ(ierr);
  ierr = IGAPointGetWorkVec(p,&G);CHKERRQ(ierr);
  ierr = Function(p,t,U,F,FunCtx);CHKERRQ(ierr);
  for (j=0; j<N; j++) {
    PetscScalar Uj = X[j];
    X[j] = Uj + h;
    ierr = Function(p,t,X,G,FunCtx);CHKERRQ(ierr);
    for (i=0; i<M; i++) J[i*N+j] = (G[i]-F[i])/h;
    X[j] = Uj;
  }
  PetscFunctionReturn(0);
}
