#include "petiga.h"

/*@
   IGAComputeScalar - Evaluates a functional of a given vector

   Collective on IGA

   Input Parameters:
+  iga - the IGA context
.  vecU - the vector to be used in computing the scalars (may be NULL)
.  n - the number of scalars to compute
.  Scalar - the function which evaluates the functional
-  ctx - user-defined context for evaluation routine (may be NULL)

   Output Parameter:
.  S - an array [0:n-1] of scalars produced by Scalar

   Details of Scalar:
$  PetscErrorCode Scalar(IGAPoint p,const PetscScalar *U,PetscInt n,PetscScalar *S,void *ctx);
+  p - point at which to evaluate the functional
.  U - the vector
.  n - the number of scalars being computed
.  S - an output array[0:n-1] of scalars
-  ctx - [optional] user-defined context for evaluation routine

   Notes:
   This function can be used to evaluate functionals of the
   solution. Use this when you wish to compute errors in the energy
   norm or moments of the solution.

   Level: normal

.keywords: IGA, evaluating functionals
@*/
PetscErrorCode IGAComputeScalar(IGA iga,Vec vecU,
                                PetscInt n,PetscScalar S[],
                                IGAFormScalar Scalar,void *ctx)
{
  MPI_Comm          comm;
  Vec               localU;
  const PetscScalar *arrayU = NULL;
  PetscScalar       *localS,*workS;
  IGAElement        element;
  IGAPoint          point;
  PetscScalar       *U;
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  if (vecU) PetscValidHeaderSpecific(vecU,VEC_CLASSID,2);
  PetscValidLogicalCollectiveInt(iga,n,3);
  PetscValidScalarPointer(S,4);
  IGACheckSetUp(iga,1);

  ierr = PetscCalloc1((size_t)n,&localS);CHKERRQ(ierr);
  ierr = PetscMalloc1((size_t)n,&workS);CHKERRQ(ierr);

  /* Get local vector U and array */
  if (vecU) {ierr = IGAGetLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);}

  ierr = PetscLogEventBegin(IGA_FormScalar,iga,vecU,0,0);CHKERRQ(ierr);

  /* Element loop */
  ierr = IGABeginElement(iga,&element);CHKERRQ(ierr);
  while (IGANextElement(iga,element)) {
    ierr = IGAElementGetValues(element,arrayU,&U);CHKERRQ(ierr);
    /* Quadrature loop */
    ierr = IGAElementBeginPoint(element,&point);CHKERRQ(ierr);
    while (IGAElementNextPoint(element,point)) {
      ierr = PetscMemzero(workS,(size_t)n*sizeof(PetscScalar));CHKERRQ(ierr);
      ierr = Scalar(point,U,n,workS,ctx);CHKERRQ(ierr);
      ierr = IGAPointAddArray(point,n,workS,localS);CHKERRQ(ierr);
    }
    ierr = IGAElementEndPoint(element,&point);CHKERRQ(ierr);
  }
  ierr = IGAEndElement(iga,&element);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(IGA_FormScalar,iga,vecU,0,0);CHKERRQ(ierr);

  /* Restore local vector U and array */
  if (vecU) {ierr = IGARestoreLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);}

  /* Assemble global scalars S */
  ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(localS,S,(PetscMPIInt)n,MPIU_SCALAR,MPIU_SUM,comm);CHKERRQ(ierr);

  ierr = PetscFree(localS);CHKERRQ(ierr);
  ierr = PetscFree(workS);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


typedef struct {
  PetscInt     order;
  IGAFormExact Exact;
  void         *ctx;
  PetscScalar  *work[2];
} ErrorCtx;

static const size_t intpow[4][5] = {{1,0,0,0,0},{1,1,1,1,1},{1,2,4,8,16},{1,3,9,27,81}};

static PetscErrorCode ErrorSqr(IGAPoint p,const PetscScalar U[],PetscInt dof,PetscScalar errsqr[],void *ctx)
{
  ErrorCtx       *app = (ErrorCtx*)ctx;
  PetscInt       dim = PetscClipInterval(p->dim,1,3);
  PetscInt       order = PetscClipInterval(app->order,0,4);
  PetscInt       i,j,n = (PetscInt)intpow[dim][order];
  PetscScalar    *v_approx = app->work[0];
  PetscScalar    *v_exact  = app->work[1];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = IGAPointEvaluate(p,order,U,v_approx);CHKERRQ(ierr);
  ierr = PetscMemzero(v_exact,(size_t)(dof*n)*sizeof(PetscScalar));CHKERRQ(ierr);
  if (app->Exact) {ierr = app->Exact(p,order,v_exact,app->ctx);CHKERRQ(ierr);}
  for (i=0; i<dof; i++) {
    PetscScalar *u = &v_exact[i*n], *uh = &v_approx[i*n];
    for (j=0; j<n; j++) { PetscReal e = PetscAbsScalar(u[j] - uh[j]); errsqr[i] += e*e; }
  }
  PetscFunctionReturn(0);
}

/*@
   IGAComputeErrorNorm - Evaluates L2 norms and H^k seminorms.

   Collective on IGA

   Input Parameters:
+  iga - the IGA context
.  k - derivative index of the error norm to compute
.  vecU - the vector to be used in computing the discrete approximation (may be NULL)
.  Exact - the function which computes exact values and gradients (may be NULL)
-  ctx - user-defined context for evaluation routine (may be NULL)

   Output Parameters:
.  enorm - an array[dof] of reals with error norms for each field

   Details of Exact:
$  PetscErrorCode Exact(IGAPoint p,PetscInt k,PetscScalar V[],void *ctx);
+  p - point at which to evaluate values or derivatives
.  k - derivative index
.  V - an output array[dof][dim**k] to store exact values
-  ctx - [optional] user-defined context for evaluation routine

   Notes:
   This function can be used to evaluate L2 or H1 errors of a discrete
   solution.
   If vecU (resp. Exact) is NULL or zero, this actually computes
   the L2 norm and H^k seminorms of the exact (resp. discrete) solution.

   Level: normal

.keywords: IGA, evaluating errors
@*/
PetscErrorCode IGAComputeErrorNorm(IGA iga,PetscInt k,
                                   Vec vecU,IGAFormExact Exact,
                                   PetscReal enorm[],void *ctx)
{
  ErrorCtx       app;
  PetscInt       i,dof,dim;
  PetscScalar    *errsqr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidLogicalCollectiveInt(iga,k,2);
  if (vecU) PetscValidHeaderSpecific(vecU,VEC_CLASSID,3);
  PetscValidRealPointer(enorm,5);
  IGACheckSetUp(iga,1);
  if (k < 0) SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Derivative index must be nonnegative, got %d",(int)k);

  ierr = IGAGetDof(iga,&dof);CHKERRQ(ierr);
  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
  ierr = PetscMalloc1((size_t)dof,&errsqr);CHKERRQ(ierr);
  ierr = PetscMalloc1((size_t)dof*intpow[dim][k],&app.work[0]);CHKERRQ(ierr);
  ierr = PetscMalloc1((size_t)dof*intpow[dim][k],&app.work[1]);CHKERRQ(ierr);
  
  app.order = k; app.Exact = Exact; app.ctx = ctx;
  ierr = IGAComputeScalar(iga,vecU,dof,errsqr,ErrorSqr,&app);CHKERRQ(ierr);
  for (i=0; i<dof; i++) enorm[i] = PetscSqrtReal(PetscRealPart(errsqr[i]));

  ierr = PetscFree(errsqr);CHKERRQ(ierr);
  ierr = PetscFree(app.work[0]);CHKERRQ(ierr);
  ierr = PetscFree(app.work[1]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
