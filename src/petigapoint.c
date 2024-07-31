#include "petiga.h"

PetscErrorCode IGAPointCreate(IGAPoint *_point)
{
  IGAPoint       point;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscAssertPointer(_point,3);
  ierr = PetscCalloc1(1,&point);CHKERRQ(ierr);
  *_point = point; point->refct =  1;
  point->index = -1;
  PetscFunctionReturn(0);
}

PetscErrorCode IGAPointDestroy(IGAPoint *_point)
{
  IGAPoint       point;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscAssertPointer(_point,1);
  point = *_point; *_point = NULL;
  if (!point) PetscFunctionReturn(0);
  if (--point->refct > 0) PetscFunctionReturn(0);
  ierr = IGAPointReset(point);CHKERRQ(ierr);
  ierr = PetscFree(point);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGAPointReference(IGAPoint point)
{
  PetscFunctionBegin;
  PetscAssertPointer(point,1);
  point->refct++;
  PetscFunctionReturn(0);
}

static
PetscErrorCode IGAPointFreeWork(IGAPoint point)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscAssertPointer(point,1);
  {
    size_t MAX_WORK_VEC = sizeof(point->wvec)/sizeof(PetscScalar*);
    size_t MAX_WORK_MAT = sizeof(point->wmat)/sizeof(PetscScalar*);
    size_t i;
    for (i=0; i<MAX_WORK_VEC; i++)
      {ierr = PetscFree(point->wvec[i]);CHKERRQ(ierr);}
    point->nvec = 0;
    for (i=0; i<MAX_WORK_MAT; i++)
      {ierr = PetscFree(point->wmat[i]);CHKERRQ(ierr);}
    point->nmat = 0;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IGAPointReset(IGAPoint point)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!point) PetscFunctionReturn(0);
  PetscAssertPointer(point,1);
  point->count =  0;
  point->index = -1;
  point->rational = NULL;
  point->geometry = NULL;
  point->property = NULL;
  ierr = IGAPointFreeWork(point);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGAPointInit(IGAPoint point,IGAElement element)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscAssertPointer(point,1);
  PetscAssertPointer(element,2);
  ierr = IGAPointReset(point);CHKERRQ(ierr);
  point->parent = element;
  point->neq = element->neq;
  point->nen = element->nen;
  point->dof = element->dof;
  point->dim = element->dim;
  point->nsd = element->nsd;
  point->npd = element->npd;
  point->rational = element->rational ? element->rationalW : NULL;
  point->geometry = element->geometry ? element->geometryX : NULL;
  point->property = element->property ? element->propertyA : NULL;
  { /* */
    size_t MAX_WORK_VEC = sizeof(point->wvec)/sizeof(PetscScalar*);
    size_t MAX_WORK_MAT = sizeof(point->wmat)/sizeof(PetscScalar*);
    size_t i, nv = (size_t)(element->nen * element->dof), nm = nv*nv;
    for (i=0; i<MAX_WORK_VEC; i++)
      {ierr = PetscMalloc1(nv,&point->wvec[i]);CHKERRQ(ierr);}
    for (i=0; i<MAX_WORK_MAT; i++)
      {ierr = PetscMalloc1(nm,&point->wmat[i]);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IGAPointGetParent(IGAPoint point,IGAElement *element)
{
  PetscFunctionBegin;
  PetscAssertPointer(point,1);
  PetscAssertPointer(element,2);
  *element = point->parent;
  PetscFunctionReturn(0);
}

PetscErrorCode IGAPointGetIndex(IGAPoint point,PetscInt *index)
{
  PetscFunctionBegin;
  PetscAssertPointer(point,1);
  PetscAssertPointer(index,2);
  *index = point->index;
  PetscFunctionReturn(0);
}

PetscErrorCode IGAPointGetCount(IGAPoint point,PetscInt *count)
{
  PetscFunctionBegin;
  PetscAssertPointer(point,1);
  PetscAssertPointer(count,2);
  *count = point->count;
  PetscFunctionReturn(0);
}

PetscErrorCode IGAPointAtBoundary(IGAPoint point,PetscInt *axis,PetscInt *side)
{
  PetscFunctionBegin;
  PetscAssertPointer(point,1);
  if (axis) PetscAssertPointer(axis,2);
  if (side) PetscAssertPointer(side,3);
  if (axis) *axis = point->atboundary ? point->boundary_id / 2 : -1;
  if (side) *side = point->atboundary ? point->boundary_id % 2 : -1;
  PetscFunctionReturn(0);
}

PetscErrorCode IGAPointGetSizes(IGAPoint point,PetscInt *neq,PetscInt *nen,PetscInt *dof)
{
  PetscFunctionBegin;
  PetscAssertPointer(point,1);
  if (neq) PetscAssertPointer(neq,2);
  if (nen) PetscAssertPointer(nen,3);
  if (dof) PetscAssertPointer(dof,4);
  if (neq) *neq = point->neq;
  if (nen) *nen = point->nen;
  if (dof) *dof = point->dof;
  PetscFunctionReturn(0);
}

PetscErrorCode IGAPointGetDims(IGAPoint point,PetscInt *dim,PetscInt *nsd,PetscInt *npd)
{
  PetscFunctionBegin;
  PetscAssertPointer(point,1);
  if (dim) PetscAssertPointer(dim,2);
  if (nsd) PetscAssertPointer(nsd,3);
  if (npd) PetscAssertPointer(npd,4);
  if (dim) *dim = point->dim;
  if (nsd) *nsd = point->nsd;
  if (npd) *npd = point->npd;
  PetscFunctionReturn(0);
}

PetscErrorCode IGAPointGetQuadrature(IGAPoint point,
                                     PetscReal *weight,
                                     PetscReal *detJac)
{
  PetscFunctionBegin;
  PetscAssertPointer(point,1);
  if (weight) PetscAssertPointer(weight,3);
  if (detJac) PetscAssertPointer(detJac,4);
  if (weight) *weight = point->weight[0];
  if (detJac) *detJac = point->detJac[0];
  PetscFunctionReturn(0);
}

PetscErrorCode IGAPointGetBasisFuns(IGAPoint point,PetscInt der,const PetscReal *basisfuns[])
{
  PetscFunctionBegin;
  PetscAssertPointer(point,1);
  PetscAssertPointer(basisfuns,3);
  if (PetscUnlikely(der < 0 || der >= (PetscInt)(sizeof(point->basis)/sizeof(PetscReal*))))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Requested derivative must be in range [0,%d], got %d",(int)(sizeof(point->basis)/sizeof(PetscReal*)-1),(int)der);
  *basisfuns = point->basis[der];
  PetscFunctionReturn(0);
}

PetscErrorCode IGAPointGetShapeFuns(IGAPoint point,PetscInt der,const PetscReal *shapefuns[])
{
  PetscFunctionBegin;
  PetscAssertPointer(point,1);
  PetscAssertPointer(shapefuns,3);
  if (PetscUnlikely(der < 0 || der >= (PetscInt)(sizeof(point->shape)/sizeof(PetscReal*))))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Requested derivative must be in range [0,%d], got %d",(int)(sizeof(point->shape)/sizeof(PetscReal*)-1),(int)der);
  *shapefuns = point->shape[der];
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
extern void IGA_GetGeomMap        (PetscInt nen,PetscInt nsd,
                                   const PetscReal N[],const PetscReal C[],PetscReal X[]);
extern void IGA_GetGradGeomMap    (PetscInt nen,PetscInt nsd,PetscInt dim,
                                   const PetscReal N[],const PetscReal C[],PetscReal F[]);
extern void IGA_GetInvGradGeomMap (PetscInt nen,PetscInt nsd,PetscInt dim,
                                   const PetscReal N[],const PetscReal C[],PetscReal G[]);
EXTERN_C_END

static PetscErrorCode IGAPointFormScale(IGAPoint p,PetscReal L[])
{
  PetscFunctionBegin;
  PetscAssertPointer(p,1);
  PetscAssertPointer(L,2);
  {
    PetscInt i;
    PetscInt dim = p->dim;
    PetscInt *ID = p->parent->ID;
    IGABasis *BD = p->parent->parent->basis;
    for (i=0; i<dim; i++)
      L[i] = BD[i]->detJac[ID[i]];
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IGAPointFormGeomMap(IGAPoint p,PetscReal x[])
{
  PetscFunctionBegin;
  PetscAssertPointer(p,1);
  PetscAssertPointer(x,2);
  if (p->geometry) {
    PetscInt i,nsd = p->nsd;
    const PetscReal *X = p->mapX[0];
    for (i=0; i<nsd; i++) x[i] = X[i];
  } else {
    PetscInt i,dim = p->dim;
    const PetscReal *X = p->mapU[0];
    for (i=0; i<dim; i++) x[i] = X[i];
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IGAPointFormGradGeomMap(IGAPoint p,PetscReal F[])
{
  PetscReal L[3] = {1,1,1};
  PetscFunctionBegin;
  PetscAssertPointer(p,1);
  PetscAssertPointer(F,2);
  (void)IGAPointFormScale(p,L);
  if (p->geometry) {
    PetscInt a,dim = p->dim;
    PetscInt i,nsd = p->nsd;
    if (dim == nsd) {
      (void)PetscMemcpy(F,p->mapX[1],(size_t)(nsd*dim)*sizeof(PetscReal));
    } else {
      const PetscReal *X = p->geometry;
      IGA_GetGradGeomMap(p->nen,nsd,dim,p->basis[1],X,F);
    }
    for (i=0; i<nsd; i++)
      for (a=0; a<dim; a++)
        F[i*dim+a] *= L[a];
  } else {
    PetscInt i,dim = p->dim;
    (void)PetscMemzero(F,(size_t)(dim*dim)*sizeof(PetscReal));
    for (i=0; i<dim; i++) F[i*(dim+1)] = L[i];
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IGAPointFormInvGradGeomMap(IGAPoint p,PetscReal G[])
{
  PetscReal L[3] = {1,1,1};
  PetscFunctionBegin;
  PetscAssertPointer(p,1);
  PetscAssertPointer(G,2);
  (void)IGAPointFormScale(p,L);
  if (p->geometry) {
    PetscInt a,dim = p->dim;
    PetscInt i,nsd = p->nsd;
    if (dim == nsd) {
      (void)PetscMemcpy(G,p->mapU[1],(size_t)(dim*nsd)*sizeof(PetscReal));
    } else {
      const PetscReal *X = p->geometry;
      IGA_GetInvGradGeomMap(p->nen,nsd,dim,p->basis[1],X,G);
    }
    for (a=0; a<dim; a++)
      for (i=0; i<nsd; i++)
        G[a*nsd+i] /= L[a];
  } else {
    PetscInt i,dim = p->dim;
    (void)PetscMemzero(G,(size_t)(dim*dim)*sizeof(PetscReal));
    for (i=0; i<dim; i++) G[i*(dim+1)] = 1/L[i];
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
extern void igapoint_geommap       (IGAPoint,PetscReal[]);
extern void igapoint_gradgeommap   (IGAPoint,PetscReal[]);
extern void igapoint_invgradgeommap(IGAPoint,PetscReal[]);
EXTERN_C_END

EXTERN_C_BEGIN
void igapoint_geommap       (IGAPoint p,PetscReal x[]) {(void)IGAPointFormGeomMap(p,x);}
void igapoint_gradgeommap   (IGAPoint p,PetscReal f[]) {(void)IGAPointFormGradGeomMap(p,f);}
void igapoint_invgradgeommap(IGAPoint p,PetscReal g[]) {(void)IGAPointFormInvGradGeomMap(p,g);}
EXTERN_C_END


EXTERN_C_BEGIN
extern void IGA_GetValue(PetscInt nen,PetscInt dof,const PetscReal N[],
                         const PetscScalar U[],PetscScalar u[]);
extern void IGA_GetGrad (PetscInt nen,PetscInt dof,PetscInt dim,const PetscReal N[],
                         const PetscScalar U[],PetscScalar u[]);
extern void IGA_GetHess (PetscInt nen,PetscInt dof,PetscInt dim,const PetscReal N[],
                         const PetscScalar U[],PetscScalar u[]);
extern void IGA_GetDel2 (PetscInt nen,PetscInt dof,PetscInt dim,const PetscReal N[],
                         const PetscScalar U[],PetscScalar u[]);
extern void IGA_GetDer3 (PetscInt nen,PetscInt dof,PetscInt dim,const PetscReal N[],
                         const PetscScalar U[],PetscScalar u[]);
extern void IGA_GetDer4 (PetscInt nen,PetscInt dof,PetscInt dim,const PetscReal N[],
                         const PetscScalar U[],PetscScalar u[]);
EXTERN_C_END

PetscErrorCode IGAPointFormPoint(IGAPoint p,PetscReal x[])
{ return IGAPointFormGeomMap(p,x); }

PetscErrorCode IGAPointFormValue(IGAPoint p,const PetscScalar U[],PetscScalar u[])
{
  PetscFunctionBegin;
  PetscAssertPointer(p,1);
  PetscAssertPointer(U,2);
  PetscAssertPointer(u,3);
  IGA_GetValue(p->nen,p->dof,p->shape[0],U,u);
  PetscFunctionReturn(0);
}

PetscErrorCode IGAPointFormGrad(IGAPoint p,const PetscScalar U[],PetscScalar u[])
{
  PetscFunctionBegin;
  PetscAssertPointer(p,1);
  PetscAssertPointer(U,2);
  PetscAssertPointer(u,3);
  IGA_GetGrad(p->nen,p->dof,p->dim,p->shape[1],U,u);
  PetscFunctionReturn(0);
}

PetscErrorCode IGAPointFormHess(IGAPoint p,const PetscScalar U[],PetscScalar u[])
{
  PetscFunctionBegin;
  PetscAssertPointer(p,1);
  PetscAssertPointer(U,2);
  PetscAssertPointer(u,3);
  IGA_GetHess(p->nen,p->dof,p->dim,p->shape[2],U,u);
  PetscFunctionReturn(0);
}

PetscErrorCode IGAPointFormDel2(IGAPoint p,const PetscScalar U[],PetscScalar u[])
{
  PetscFunctionBegin;
  PetscAssertPointer(p,1);
  PetscAssertPointer(U,2);
  PetscAssertPointer(u,3);
  IGA_GetDel2(p->nen,p->dof,p->dim,p->shape[2],U,u);
  PetscFunctionReturn(0);
}

PetscErrorCode IGAPointFormDer3(IGAPoint p,const PetscScalar U[],PetscScalar u[])
{
  PetscFunctionBegin;
  PetscAssertPointer(p,1);
  PetscAssertPointer(U,2);
  PetscAssertPointer(u,3);
  IGA_GetDer3(p->nen,p->dof,p->dim,p->shape[3],U,u);
  PetscFunctionReturn(0);
}

PetscErrorCode IGAPointFormDer4(IGAPoint p,const PetscScalar U[],PetscScalar u[])
{
  PetscFunctionBegin;
  PetscAssertPointer(p,1);
  PetscAssertPointer(U,2);
  PetscAssertPointer(u,3);
  IGA_GetDer4(p->nen,p->dof,p->dim,p->shape[4],U,u);
  PetscFunctionReturn(0);
}

PetscErrorCode IGAPointEvaluate(IGAPoint p,PetscInt ider,const PetscScalar U[],PetscScalar u[])
{
  PetscFunctionBegin;
  PetscAssertPointer(p,1);
  PetscAssertPointer(U,4);
  PetscAssertPointer(u,4);
  if (PetscUnlikely(p->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during point loop");
  if (PetscUnlikely(ider < 0 || ider > 4))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Expecting 0<=ider<=4, got %d",(int)ider);
  {
    PetscInt nen = p->nen;
    PetscInt dof = p->dof;
    PetscInt dim = p->dim;
    PetscReal *N = p->shape[ider];
    switch (ider) {
    case 0: IGA_GetValue(nen,dof,/**/N,U,u); break;
    case 1: IGA_GetGrad (nen,dof,dim,N,U,u); break;
    case 2: IGA_GetHess (nen,dof,dim,N,U,u); break;
    case 3: IGA_GetDer3 (nen,dof,dim,N,U,u); break;
    case 4: IGA_GetDer4 (nen,dof,dim,N,U,u); break;
    default: PetscFunctionReturn(PETSC_ERR_ARG_OUTOFRANGE);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IGAPointGetWorkVec(IGAPoint point,PetscScalar *V[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscAssertPointer(point,1);
  PetscAssertPointer(V,2);
  if (PetscUnlikely(point->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during point loop");
  if (PetscUnlikely((size_t)point->nvec >= sizeof(point->wvec)/sizeof(PetscScalar*)))
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many work vectors requested");
  {
    size_t m = (size_t)(point->neq * point->dof);
    *V = point->wvec[point->nvec++];
    ierr = PetscMemzero(*V,(size_t)m*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IGAPointGetWorkMat(IGAPoint point,PetscScalar *M[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscAssertPointer(point,1);
  PetscAssertPointer(M,2);
  if (PetscUnlikely(point->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during point loop");
  if (PetscUnlikely((size_t)point->nmat >= sizeof(point->wmat)/sizeof(PetscScalar*)))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many work matrices requested");
  {
    size_t m = (size_t)(point->neq * point->dof);
    size_t n = (size_t)(point->nen * point->dof);
    *M = point->wmat[point->nmat++];
    ierr = PetscMemzero(*M,(size_t)(m*n)*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IGAPointAddArray(IGAPoint point,PetscInt n,const PetscScalar a[],PetscScalar A[])
{
  PetscInt  i;
  PetscReal JW;
  PetscFunctionBegin;
  PetscAssertPointer(point,1);
  PetscAssertPointer(a,3);
  PetscAssertPointer(A,4);
  if (PetscUnlikely(point->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during point loop");
  JW = point->detJac[0] * point->weight[0];
  for (i=0; i<n; i++) A[i] += a[i] * JW;
  (void)PetscLogFlops((PetscLogDouble)(2*n));
  PetscFunctionReturn(0);
}

PetscErrorCode IGAPointAddVec(IGAPoint point,const PetscScalar f[],PetscScalar F[])
{
  PetscInt       m;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscAssertPointer(point,1);
  PetscAssertPointer(f,2);
  PetscAssertPointer(F,3);
  m = point->neq * point->dof;
  ierr = IGAPointAddArray(point,m,f,F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGAPointAddMat(IGAPoint point,const PetscScalar k[],PetscScalar K[])
{
  PetscInt       m,n;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscAssertPointer(point,1);
  PetscAssertPointer(k,2);
  PetscAssertPointer(K,3);
  m = point->neq * point->dof;
  n = point->nen * point->dof;
  ierr = IGAPointAddArray(point,m*n,k,K);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
