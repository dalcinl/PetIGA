#include "petiga.h"

EXTERN_C_BEGIN
extern PetscInt IGA_FindSpan(PetscInt n,PetscInt p,PetscReal u,const PetscReal U[]);
EXTERN_C_END

EXTERN_C_BEGIN
extern void IGA_Basis_BSpline (PetscInt i,PetscReal u,PetscInt p,PetscInt d,const PetscReal U[],PetscReal B[]);
extern void IGA_Basis_Lagrange(PetscInt i,PetscReal u,PetscInt p,PetscInt d,const PetscReal U[],PetscReal L[]);
extern void IGA_Basis_Spectral(PetscInt i,PetscReal u,PetscInt p,PetscInt d,const PetscReal U[],PetscReal L[]);
EXTERN_C_END

#include "petigaftn.h"

EXTERN_C_BEGIN
extern void IGA_GetValue(PetscInt nen,PetscInt dof,/*         */const PetscReal N[],const PetscScalar U[],PetscScalar u[]);
extern void IGA_GetGrad (PetscInt nen,PetscInt dof,PetscInt dim,const PetscReal N[],const PetscScalar U[],PetscScalar u[]);
extern void IGA_GetHess (PetscInt nen,PetscInt dof,PetscInt dim,const PetscReal N[],const PetscScalar U[],PetscScalar u[]);
extern void IGA_GetDer3 (PetscInt nen,PetscInt dof,PetscInt dim,const PetscReal N[],const PetscScalar U[],PetscScalar u[]);
extern void IGA_GetDer4 (PetscInt nen,PetscInt dof,PetscInt dim,const PetscReal N[],const PetscScalar U[],PetscScalar u[]);
EXTERN_C_END

PetscErrorCode IGAProbeCreate(IGA iga,Vec A,IGAProbe *_prb)
{
  PetscInt       i;
  IGAProbe       prb;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  if (A) PetscValidHeaderSpecific(A,VEC_CLASSID,2);
  PetscValidPointer(_prb,3);
  IGACheckSetUp(iga,1);

  ierr = PetscCalloc1(1,&prb);CHKERRQ(ierr);
  *_prb = prb; prb->refct = 1;

  ierr = PetscObjectReference((PetscObject)iga);CHKERRQ(ierr);
  prb->iga  = iga;

  ierr = IGAGetDim(iga,&prb->dim);CHKERRQ(ierr);
  prb->nsd = iga->geometry ? iga->geometry : iga->dim;
  ierr = IGAGetDof(iga,&prb->dof);CHKERRQ(ierr);

  for (i=0; i<3; i++) {
    prb->p[i] = iga->axis[i]->p;
    prb->U[i] = iga->axis[i]->U;
    prb->n[i] = iga->geom_sizes[i];
    prb->s[i] = iga->geom_gstart[i];
    prb->w[i] = iga->geom_gwidth[i];
  }
  prb->nen = 1;
  for (i=0; i<3; i++) {
    PetscInt n = prb->p[i]+1;
    ierr = PetscMalloc1((size_t)(n*5),&prb->BD[i]);CHKERRQ(ierr);
    prb->nen *= n;
  }
  if (iga->rational && iga->rationalW)
    prb->arrayW = iga->rationalW;
  if (iga->geometry && iga->geometryX)
    prb->arrayX = iga->geometryX;

  {
    size_t nen = (size_t)prb->nen;
    size_t dim = (size_t)prb->dim;
    size_t nsd = (size_t)prb->nsd;
    size_t dof = (size_t)prb->dof;

    ierr = PetscCalloc1(nen,&prb->map);CHKERRQ(ierr);
    ierr = PetscCalloc1(nen,&prb->W);CHKERRQ(ierr);
    ierr = PetscCalloc1(nen*nsd,&prb->X);CHKERRQ(ierr);
    ierr = PetscCalloc1(nen*dof,&prb->A);CHKERRQ(ierr);

    ierr = PetscCalloc1(nen,&prb->basis[0]);CHKERRQ(ierr);
    ierr = PetscCalloc1(nen*dim,&prb->basis[1]);CHKERRQ(ierr);
    ierr = PetscCalloc1(nen*dim*dim,&prb->basis[2]);CHKERRQ(ierr);
    ierr = PetscCalloc1(nen*dim*dim*dim,&prb->basis[3]);CHKERRQ(ierr);
    ierr = PetscCalloc1(nen*dim*dim*dim*dim,&prb->basis[4]);CHKERRQ(ierr);

    ierr = PetscCalloc1(nen,&prb->shape[0]);CHKERRQ(ierr);
    ierr = PetscCalloc1(nen*nsd,&prb->shape[1]);CHKERRQ(ierr);
    ierr = PetscCalloc1(nen*nsd*nsd,&prb->shape[2]);CHKERRQ(ierr);
    ierr = PetscCalloc1(nen*nsd*nsd*nsd,&prb->shape[3]);CHKERRQ(ierr);
    ierr = PetscCalloc1(nen*nsd*nsd*nsd*nsd,&prb->shape[4]);CHKERRQ(ierr);

    ierr = PetscCalloc1(dim,&prb->mapU[0]);CHKERRQ(ierr);
    ierr = PetscCalloc1(dim*nsd,&prb->mapU[1]);CHKERRQ(ierr);
    ierr = PetscCalloc1(dim*nsd*nsd,&prb->mapU[2]);CHKERRQ(ierr);
    ierr = PetscCalloc1(dim*nsd*nsd*nsd,&prb->mapU[3]);CHKERRQ(ierr);
    ierr = PetscCalloc1(dim*nsd*nsd*nsd*nsd,&prb->mapU[4]);CHKERRQ(ierr);

    ierr = PetscCalloc1(nsd,&prb->mapX[0]);CHKERRQ(ierr);
    ierr = PetscCalloc1(nsd*dim,&prb->mapX[1]);CHKERRQ(ierr);
    ierr = PetscCalloc1(nsd*dim*dim,&prb->mapX[2]);CHKERRQ(ierr);
    ierr = PetscCalloc1(nsd*dim*dim*dim,&prb->mapX[3]);CHKERRQ(ierr);
    ierr = PetscCalloc1(nsd*dim*dim*dim*dim,&prb->mapX[4]);CHKERRQ(ierr);

    ierr = PetscCalloc1(1,&prb->detX);CHKERRQ(ierr);
  }
  {
    PetscInt dim  = prb->dim;
    PetscInt nsd  = prb->nsd;
    PetscReal *dX = prb->detX;
    PetscReal *X1 = prb->mapX[1];
    PetscReal *U1 = prb->mapU[1];
    dX[0] = 1.0;
    for (i=0; i<dim; i++) X1[i*(dim+1)] = 1.0;
    for (i=0; i<dim; i++) U1[i*(nsd+1)] = 1.0;
  }

  ierr = IGAGetOrder(iga,&prb->order);CHKERRQ(ierr);
  ierr = IGAProbeSetOrder(prb,prb->order);CHKERRQ(ierr);
  ierr = IGAProbeSetCollective(prb,PETSC_TRUE);CHKERRQ(ierr);

  for (i=0; i<prb->dim; i++) prb->point[i] = prb->U[i][prb->p[i]];
  if (A) {ierr = IGAProbeSetVec(prb,A);CHKERRQ(ierr);}

  PetscFunctionReturn(0);
}

PetscErrorCode IGAProbeDestroy(IGAProbe *_prb)
{
  IGAProbe       prb;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(_prb,1);
  prb = *_prb; *_prb = NULL;
  if (!prb) PetscFunctionReturn(0);
  if (--prb->refct > 0) PetscFunctionReturn(0);

  if (prb->gvec) {ierr = IGARestoreLocalVecArray(prb->iga,prb->gvec,&prb->lvec,&prb->arrayA);CHKERRQ(ierr);}
  ierr = VecDestroy(&prb->gvec);CHKERRQ(ierr);
  ierr = IGADestroy(&prb->iga);CHKERRQ(ierr);

  ierr = PetscFree(prb->BD[0]);CHKERRQ(ierr);
  ierr = PetscFree(prb->BD[1]);CHKERRQ(ierr);
  ierr = PetscFree(prb->BD[2]);CHKERRQ(ierr);

  ierr = PetscFree(prb->map);CHKERRQ(ierr);
  ierr = PetscFree(prb->W);CHKERRQ(ierr);
  ierr = PetscFree(prb->X);CHKERRQ(ierr);
  ierr = PetscFree(prb->A);CHKERRQ(ierr);

  ierr = PetscFree(prb->basis[0]);CHKERRQ(ierr);
  ierr = PetscFree(prb->basis[1]);CHKERRQ(ierr);
  ierr = PetscFree(prb->basis[2]);CHKERRQ(ierr);
  ierr = PetscFree(prb->basis[3]);CHKERRQ(ierr);
  ierr = PetscFree(prb->basis[4]);CHKERRQ(ierr);

  ierr = PetscFree(prb->detX);CHKERRQ(ierr);

  ierr = PetscFree(prb->mapX[0]);CHKERRQ(ierr);
  ierr = PetscFree(prb->mapX[1]);CHKERRQ(ierr);
  ierr = PetscFree(prb->mapX[2]);CHKERRQ(ierr);
  ierr = PetscFree(prb->mapX[3]);CHKERRQ(ierr);
  ierr = PetscFree(prb->mapX[4]);CHKERRQ(ierr);

  ierr = PetscFree(prb->mapU[0]);CHKERRQ(ierr);
  ierr = PetscFree(prb->mapU[1]);CHKERRQ(ierr);
  ierr = PetscFree(prb->mapU[2]);CHKERRQ(ierr);
  ierr = PetscFree(prb->mapU[3]);CHKERRQ(ierr);
  ierr = PetscFree(prb->mapU[4]);CHKERRQ(ierr);

  ierr = PetscFree(prb->shape[0]);CHKERRQ(ierr);
  ierr = PetscFree(prb->shape[1]);CHKERRQ(ierr);
  ierr = PetscFree(prb->shape[2]);CHKERRQ(ierr);
  ierr = PetscFree(prb->shape[3]);CHKERRQ(ierr);
  ierr = PetscFree(prb->shape[4]);CHKERRQ(ierr);

  ierr = PetscFree(prb);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGAProbeReference(IGAProbe prb)
{
  PetscFunctionBegin;
  PetscValidPointer(prb,1);
  prb->refct++;
  PetscFunctionReturn(0);
}

PetscErrorCode IGAProbeSetOrder(IGAProbe prb,PetscInt order)
{
  PetscFunctionBegin;
  PetscValidPointer(prb,1);
  PetscValidLogicalCollectiveInt(prb->iga,order,2);
  if (PetscUnlikely(order < 0 || order > 4)) SETERRQ1(((PetscObject)prb->iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Expecting 0<=order<=4, got %D",order);
  prb->order = order;
  PetscFunctionReturn(0);
}

PetscErrorCode IGAProbeSetCollective(IGAProbe prb,PetscBool collective)
{
  MPI_Comm       comm;
  PetscMPIInt    size;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(prb,1);
  PetscValidLogicalCollectiveBool(prb->iga,collective,2);
  prb->collective = collective;
  if (prb->collective) {
    ierr = IGAGetComm(prb->iga,&comm);CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
    if (size == 1) prb->collective = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IGAProbeSetVec(IGAProbe prb,Vec A)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(prb,1);
  PetscValidHeaderSpecific(A,VEC_CLASSID,2);
  if (prb->gvec) {ierr = IGARestoreLocalVecArray(prb->iga,prb->gvec,&prb->lvec,&prb->arrayA);CHKERRQ(ierr);}
  ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
  ierr = VecDestroy(&prb->gvec);CHKERRQ(ierr);
  prb->gvec = A;
  ierr = IGAGetLocalVecArray(prb->iga,prb->gvec,&prb->lvec,&prb->arrayA);CHKERRQ(ierr);
  ierr = IGAProbeSetPoint(prb,prb->point);CHKERRQ(ierr); /* XXX refresh! */
  PetscFunctionReturn(0);
}

PetscErrorCode IGAProbeSetPoint(IGAProbe prb,const PetscReal u[])
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(prb,1);
  PetscValidRealPointer(u,2);
#if defined(PETSC_USE_DEBUG)
  for (i=0; i<prb->dim; i++) {
    PetscReal *U = prb->U[i];
    PetscInt   a = prb->p[i];
    PetscInt   b = prb->n[i];
    if (PetscUnlikely(u[i] < U[a] || u[i] > U[b]))
      SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
               "Expecting %g <= u[%D]=%g <= %g",(double)U[a],i,(double)u[i],(double)U[b]);
    if (prb->collective) PetscValidLogicalCollectiveReal(prb->iga,u[i],2);
  }
#endif
  for (i=0; i<prb->dim; i++) prb->point[i] = u[i];

  /* Span index */
  {
    for (i=0; i<prb->dim; i++)
      prb->ID[i] = IGA_FindSpan(prb->n[i]-1,prb->p[i],prb->point[i],prb->U[i]);
  }

  prb->offprocess = PETSC_FALSE;
  { /* XXX Optimize: if (comm_size > 1) */
    for (i=0; i<prb->dim; i++) {
      PetscInt first = prb->ID[i]-prb->p[i], last = first + prb->p[i];
      PetscInt start = prb->s[i], end = prb->s[i] + prb->w[i];
      if (first < start || last >= end) prb->offprocess = PETSC_TRUE;
    }
  }
  if (PetscUnlikely(prb->offprocess && !prb->collective)) PetscFunctionReturn(0);

  /* Span closure */
  if (PetscLikely(!prb->offprocess)) {
    PetscInt a,nen = prb->nen;
    PetscInt k,nsd = prb->nsd;
    PetscInt c,dof = prb->dof;
    PetscInt *map  = prb->map;
    {
      PetscInt *ID = prb->ID, *p = prb->p, *s = prb->s, *w = prb->w;
      PetscInt ia, inen = p[0]+1, ioffset = ID[0]-p[0], istart = s[0];
      PetscInt ja, jnen = p[1]+1, joffset = ID[1]-p[1], jstart = s[1];
      PetscInt ka, knen = p[2]+1, koffset = ID[2]-p[2], kstart = s[2];
      PetscInt pos = 0, jstride = w[0], kstride = w[0]*w[1];
      for (ka=0; ka<knen; ka++) {
        for (ja=0; ja<jnen; ja++) {
          for (ia=0; ia<inen; ia++) {
            PetscInt iA = (ioffset + ia) - istart;
            PetscInt jA = (joffset + ja) - jstart;
            PetscInt kA = (koffset + ka) - kstart;
            map[pos++] = iA + jA*jstride + kA*kstride;
          }
        }
      }
    }
    if (prb->arrayW)
      for (a=0; a<nen; a++)
        prb->W[a] = prb->arrayW[map[a]];
    if (prb->arrayX)
      for (a=0; a<nen; a++)
        for (k=0; k<nsd; k++)
          prb->X[k + a*nsd] = prb->arrayX[k + map[a]*nsd];
    if (prb->arrayA)
      for (a=0; a<nen; a++)
        for (c=0; c<dof; c++)
          prb->A[c + a*dof] = prb->arrayA[c + map[a]*dof];
  }
  if (PetscUnlikely(prb->collective)) {
    MPI_Comm    comm;
    PetscMPIInt nen = (PetscMPIInt)prb->nen;
    PetscMPIInt nsd = (PetscMPIInt)prb->nsd;
    PetscMPIInt dof = (PetscMPIInt)prb->dof;
    PetscMPIInt rank=-1,root;
    ierr = IGAGetComm(prb->iga,&comm);CHKERRQ(ierr);
    if (!prb->offprocess) {ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);}
    ierr = MPI_Allreduce(&rank,&root,1,MPI_INT,MPI_MAX,comm);CHKERRQ(ierr);
    if (prb->arrayW) {ierr = MPI_Bcast(prb->W,nen,MPIU_REAL,root,comm);CHKERRQ(ierr);}
    if (prb->arrayX) {ierr = MPI_Bcast(prb->X,nen*nsd,MPIU_REAL,root,comm);CHKERRQ(ierr);}
    if (prb->arrayA) {ierr = MPI_Bcast(prb->A,nen*dof,MPIU_SCALAR,root,comm);CHKERRQ(ierr);}
  }

  /* Compute 1D basis functions */
  {
    IGABasis *basis = prb->iga->basis;
    void (*ComputeBasis)(PetscInt,PetscReal,PetscInt,PetscInt,const PetscReal[],PetscReal[]) = NULL;
    for (i=0; i<prb->dim; i++) {
      switch (basis[i]->type) {
      case IGA_BASIS_BSPLINE:
      case IGA_BASIS_BERNSTEIN:
        ComputeBasis = IGA_Basis_BSpline;  break;
      case IGA_BASIS_LAGRANGE:
        ComputeBasis = IGA_Basis_Lagrange; break;
      case IGA_BASIS_SPECTRAL:
        ComputeBasis = IGA_Basis_Spectral; break;
      }
      ComputeBasis(prb->ID[i],prb->point[i],prb->p[i],prb->order,prb->U[i],prb->BD[i]);
    }
  }

  /* Tensor product 1D basis functions */
  {
    PetscReal **M = prb->basis;
    switch (prb->dim) {
    case 3: IGA_BasisFuns_3D(prb->order,
                             1,prb->p[0]+1,prb->BD[0],
                             1,prb->p[1]+1,prb->BD[1],
                             1,prb->p[2]+1,prb->BD[2],
                             M[0],M[1],M[2],M[3],M[4]); break;
    case 2: IGA_BasisFuns_2D(prb->order,
                             1,prb->p[0]+1,prb->BD[0],
                             1,prb->p[1]+1,prb->BD[1],
                             M[0],M[1],M[2],M[3],M[4]); break;
    case 1: IGA_BasisFuns_1D(prb->order,
                             1,prb->p[0]+1,prb->BD[0],
                             M[0],M[1],M[2],M[3],M[4]); break;
    }
  }

  /* Rationalize basis functions */
  if (prb->arrayW) {
    PetscReal **M = prb->basis;
    switch (prb->dim) {
    case 3: IGA_Rationalize_3D(prb->order,1,prb->nen,prb->W,
                               M[0],M[1],M[2],M[3],M[4]); break;
    case 2: IGA_Rationalize_2D(prb->order,1,prb->nen,prb->W,
                               M[0],M[1],M[2],M[3],M[4]); break;
    case 1: IGA_Rationalize_1D(prb->order,1,prb->nen,prb->W,
                               M[0],M[1],M[2],M[3],M[4]); break;
    }
  }

  /* Geometry mapping */
  for (i=0; i<prb->dim; i++) {
    prb->mapU[0][i] = prb->point[i];
    prb->mapX[0][i] = prb->point[i];
  }
  if (prb->arrayX) {
    PetscReal **M = prb->basis;
    PetscReal *X0 = prb->mapX[0];
    PetscReal *X1 = prb->mapX[1];
    PetscReal *X2 = prb->mapX[2];
    PetscReal *X3 = prb->mapX[3];
    PetscReal *X4 = prb->mapX[4];
    if (PetscLikely(prb->dim == prb->nsd))
      switch (prb->dim) {
      case 3: IGA_GeometryMap_3D(prb->order,1,prb->nen,prb->X,
                                 M[0],M[1],M[2],M[3],M[4],
                                 X0,X1,X2,X3,X4); break;
      case 2: IGA_GeometryMap_2D(prb->order,1,prb->nen,prb->X,
                                 M[0],M[1],M[2],M[3],M[4],
                                 X0,X1,X2,X3,X4); break;
      case 1: IGA_GeometryMap_1D(prb->order,1,prb->nen,prb->X,
                                 M[0],M[1],M[2],M[3],M[4],
                                 X0,X1,X2,X3,X4); break;
      }
    else
      IGA_GeometryMap(prb->order,prb->dim,prb->nsd,
                      1,prb->nen,prb->X,
                      M[0],M[1],M[2],M[3],M[4],
                      X0,X1,X2,X3,X4);
  }
  if (prb->arrayX && PetscLikely(prb->dim == prb->nsd)) {
    PetscReal **M = prb->basis;
    PetscReal **N = prb->shape;
    PetscReal *dX = prb->detX;
    PetscReal *X1 = prb->mapX[1], *U1 = prb->mapU[1];
    PetscReal *X2 = prb->mapX[2], *U2 = prb->mapU[2];
    PetscReal *X3 = prb->mapX[3], *U3 = prb->mapU[3];
    PetscReal *X4 = prb->mapX[4], *U4 = prb->mapU[4];
    switch (prb->dim) {
    case 3: IGA_InverseMap_3D(prb->order,1,
                              X1,X2,X3,X4,dX,
                              U1,U2,U3,U4); break;
    case 2: IGA_InverseMap_2D(prb->order,1,
                              X1,X2,X3,X4,dX,
                              U1,U2,U3,U4); break;
    case 1: IGA_InverseMap_1D(prb->order,1,
                              X1,X2,X3,X4,dX,
                              U1,U2,U3,U4); break;
    }
    ierr = PetscMemcpy(N[0],M[0],(size_t)prb->nen*sizeof(PetscReal));CHKERRQ(ierr);
    switch (prb->dim) {
    case 3: IGA_ShapeFuns_3D(prb->order,1,prb->nen,
                             U1,U2,U3,U4,
                             M[1],M[2],M[3],M[4],
                             N[1],N[2],N[3],N[4]); break;
    case 2: IGA_ShapeFuns_2D(prb->order,1,prb->nen,
                             U1,U2,U3,U4,
                             M[1],M[2],M[3],M[4],
                             N[1],N[2],N[3],N[4]); break;
    case 1: IGA_ShapeFuns_1D(prb->order,1,prb->nen,
                             U1,U2,U3,U4,
                             M[1],M[2],M[3],M[4],
                             N[1],N[2],N[3],N[4]); break;
    }
  }

  PetscFunctionReturn(0);
}

PetscErrorCode IGAProbeGeomMap(IGAProbe prb,PetscReal x[])
{
  PetscFunctionBegin;
  PetscValidPointer(prb,1);
  PetscValidRealPointer(x,2);
  if (PetscUnlikely(prb->offprocess && !prb->collective)) {
    PetscInt i; for (i=0; i<prb->nsd; i++) x[i] = 0.0;
  } else {
    PetscReal *X = prb->arrayX ? prb->mapX[0] : prb->mapU[0];
    PetscInt i; for (i=0; i<prb->nsd; i++) x[i] = X[i];
  }
  PetscFunctionReturn(0);
}

static const size_t intpow[4][5] = {{1,0,0,0,0},{1,1,1,1,1},{1,2,4,8,16},{1,3,9,27,81}};

PetscErrorCode IGAProbeEvaluate(IGAProbe prb,PetscInt der,PetscScalar A[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(prb,1);
  PetscValidScalarPointer(A,3);
  if (PetscUnlikely(der < 0 || der > prb->order)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Expecting 0<=der<=%D, got der=%D",prb->order,der);
  if (PetscUnlikely(!prb->arrayA)) SETERRQ(((PetscObject)prb->iga)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must call IGAProbeSetVec() first");
  if (PetscUnlikely(prb->offprocess && !prb->collective)) {
    size_t n = (size_t)prb->dof * intpow[prb->dim][der];
    ierr = PetscMemzero(A,n*sizeof(PetscScalar));CHKERRQ(ierr);
  } else {
    PetscReal **shape = (prb->arrayX && prb->dim == prb->nsd) ? prb->shape : prb->basis;
    switch (der) {
    case 0: IGA_GetValue(prb->nen,prb->dof,/*     */shape[0],prb->A,A); break;
    case 1: IGA_GetGrad (prb->nen,prb->dof,prb->dim,shape[1],prb->A,A); break;
    case 2: IGA_GetHess (prb->nen,prb->dof,prb->dim,shape[2],prb->A,A); break;
    case 3: IGA_GetDer3 (prb->nen,prb->dof,prb->dim,shape[3],prb->A,A); break;
    case 4: IGA_GetDer4 (prb->nen,prb->dof,prb->dim,shape[4],prb->A,A); break;
    default: PetscFunctionReturn(PETSC_ERR_ARG_OUTOFRANGE);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IGAProbeFormValue(IGAProbe prb,PetscScalar A[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = IGAProbeEvaluate(prb,0,A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGAProbeFormGrad(IGAProbe prb,PetscScalar A[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = IGAProbeEvaluate(prb,1,A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGAProbeFormHess(IGAProbe prb,PetscScalar A[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = IGAProbeEvaluate(prb,2,A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGAProbeFormDer3(IGAProbe prb,PetscScalar A[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = IGAProbeEvaluate(prb,3,A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGAProbeFormDer4(IGAProbe prb,PetscScalar A[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = IGAProbeEvaluate(prb,4,A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
