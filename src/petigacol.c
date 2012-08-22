#include "petiga.h"

extern PetscLogEvent IGA_ColFormSystem;

#undef  __FUNCT__
#define __FUNCT__ "IGAColPointCreate"
PetscErrorCode IGAColPointCreate(IGAColPoint *_point)
{
  IGAColPoint point;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(_point,1);
  ierr = PetscNew(struct _n_IGAColPoint,_point);CHKERRQ(ierr);
  point = *_point;
  point->refct =  1;
  point->index = -1;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAColPointDestroy"
PetscErrorCode IGAColPointDestroy(IGAColPoint *_point)
{
  IGAColPoint     point;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(_point,1);
  point = *_point; *_point = 0;
  if (!point) PetscFunctionReturn(0);
  if (--point->refct > 0) PetscFunctionReturn(0);
  ierr = IGAColPointReset(point);CHKERRQ(ierr);
  ierr = PetscFree(point);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAColPointFreeWork"
static
PetscErrorCode IGAColPointFreeWork(IGAColPoint point)
{
  size_t i;
  size_t MAX_WORK_VEC;
  size_t MAX_WORK_MAT;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  MAX_WORK_VEC = sizeof(point->wvec)/sizeof(PetscScalar*);
  for (i=0; i<MAX_WORK_VEC; i++)
    {ierr = PetscFree(point->wvec[i]);CHKERRQ(ierr);}
  point->nvec = 0;
  MAX_WORK_MAT = sizeof(point->wmat)/sizeof(PetscScalar*);
  for (i=0; i<MAX_WORK_MAT; i++)
    {ierr = PetscFree(point->wmat[i]);CHKERRQ(ierr);}
  point->nmat = 0;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAColPointReset"
PetscErrorCode IGAColPointReset(IGAColPoint point)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!point) PetscFunctionReturn(0);
  PetscValidPointer(point,1);
  point->index = -1;
  ierr = IGAColPointFreeWork(point);CHKERRQ(ierr);

  ierr = PetscFree(point->mapping);CHKERRQ(ierr);
  ierr = PetscFree(point->geometryX);CHKERRQ(ierr);
  ierr = PetscFree(point->geometryW);CHKERRQ(ierr);

  ierr = PetscFree(point->point);CHKERRQ(ierr);
  ierr = PetscFree(point->scale);CHKERRQ(ierr);
  ierr = PetscFree(point->basis1d[0]);CHKERRQ(ierr);
  ierr = PetscFree(point->basis1d[1]);CHKERRQ(ierr);
  ierr = PetscFree(point->basis1d[2]);CHKERRQ(ierr);
  ierr = PetscFree(point->basis[0]);CHKERRQ(ierr);
  ierr = PetscFree(point->basis[1]);CHKERRQ(ierr);
  ierr = PetscFree(point->basis[2]);CHKERRQ(ierr);
  ierr = PetscFree(point->basis[3]);CHKERRQ(ierr);

  ierr = PetscFree(point->gradX[0]);CHKERRQ(ierr);
  ierr = PetscFree(point->gradX[1]);CHKERRQ(ierr);
  ierr = PetscFree(point->shape[0]);CHKERRQ(ierr);
  ierr = PetscFree(point->shape[1]);CHKERRQ(ierr);
  ierr = PetscFree(point->shape[2]);CHKERRQ(ierr);
  ierr = PetscFree(point->shape[3]);CHKERRQ(ierr);
  PetscFunctionReturn(0);

}

#undef  __FUNCT__
#define __FUNCT__ "IGAColPointInit"
PetscErrorCode IGAColPointInit(IGAColPoint point,IGA iga)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  PetscValidHeaderSpecific(iga,IGA_CLASSID,0);
  if (PetscUnlikely(!iga->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call IGASetUp() first");
  ierr = IGAColPointReset(point);CHKERRQ(ierr);
  point->parent = iga;

  point->dof = iga->dof;
  point->dim = iga->dim;
  point->nsd = iga->nsd ? iga->nsd : iga->dim;

  IGABasis *BD = iga->basis;
  { /* */
    PetscInt i,dim = point->dim;
    PetscInt npts=1,nen=1;
    for (i=0; i<dim; i++) {
      point->start[i] = iga->node_lstart[i];
      point->width[i] = iga->node_lwidth[i];
      npts *= iga->node_lwidth[i];
      nen  *= BD[i]->nen;
    }
    point->index = -1;
    point->count = npts;
    point->nen   = nen;
  }
  { /* */
    PetscInt dim = point->dim;
    PetscInt nsd = point->nsd;
    PetscInt nen = point->nen;
    PetscInt nqp = 1; // looks like the element code, but just for 1 point

    ierr = PetscMalloc1(nen,PetscInt,&point->mapping);CHKERRQ(ierr);
    ierr = PetscMalloc1(nen*nsd,PetscReal,&point->geometryX);CHKERRQ(ierr);
    ierr = PetscMalloc1(nen    ,PetscReal,&point->geometryW);CHKERRQ(ierr);    

    ierr = PetscMalloc1(nqp*dim,PetscReal,&point->point);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*dim,PetscReal,&point->scale);CHKERRQ(ierr);

    ierr = PetscMalloc1(BD[0]->nen*(iga->order+1),PetscReal,&point->basis1d[0]);CHKERRQ(ierr);
    ierr = PetscMalloc1(BD[1]->nen*(iga->order+1),PetscReal,&point->basis1d[1]);CHKERRQ(ierr);
    ierr = PetscMalloc1(BD[2]->nen*(iga->order+1),PetscReal,&point->basis1d[2]);CHKERRQ(ierr);

    ierr = PetscMalloc1(nqp*nen,PetscReal,&point->basis[0]);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*nen*dim,PetscReal,&point->basis[1]);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*nen*dim*dim,PetscReal,&point->basis[2]);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*nen*dim*dim*dim,PetscReal,&point->basis[3]);CHKERRQ(ierr);

    ierr = PetscMalloc1(nqp*dim*dim,PetscReal,&point->gradX[0]);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*dim*dim,PetscReal,&point->gradX[1]);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*nen,PetscReal,&point->shape[0]);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*nen*dim,PetscReal,&point->shape[1]);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*nen*dim*dim,PetscReal,&point->shape[2]);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*nen*dim*dim*dim,PetscReal,&point->shape[3]);CHKERRQ(ierr);

    ierr = PetscMemzero(point->basis1d[0],BD[0]->nen*(iga->order+1)*sizeof(PetscReal));CHKERRQ(ierr);
    ierr = PetscMemzero(point->basis1d[1],BD[1]->nen*(iga->order+1)*sizeof(PetscReal));CHKERRQ(ierr);
    ierr = PetscMemzero(point->basis1d[2],BD[2]->nen*(iga->order+1)*sizeof(PetscReal));CHKERRQ(ierr);

    ierr = PetscMemzero(point->point,   sizeof(PetscReal)*nqp*dim);CHKERRQ(ierr);
    ierr = PetscMemzero(point->scale,   sizeof(PetscReal)*nqp*dim);CHKERRQ(ierr);
    ierr = PetscMemzero(point->basis[0],sizeof(PetscReal)*nqp*nen);CHKERRQ(ierr);
    ierr = PetscMemzero(point->basis[1],sizeof(PetscReal)*nqp*nen*dim);CHKERRQ(ierr);
    ierr = PetscMemzero(point->basis[2],sizeof(PetscReal)*nqp*nen*dim*dim);CHKERRQ(ierr);
    ierr = PetscMemzero(point->basis[3],sizeof(PetscReal)*nqp*nen*dim*dim*dim);CHKERRQ(ierr);

    ierr = PetscMemzero(point->gradX[0],sizeof(PetscReal)*nqp*dim*dim);CHKERRQ(ierr);
    ierr = PetscMemzero(point->gradX[1],sizeof(PetscReal)*nqp*dim*dim);CHKERRQ(ierr);
    ierr = PetscMemzero(point->shape[0],sizeof(PetscReal)*nqp*nen);CHKERRQ(ierr);
    ierr = PetscMemzero(point->shape[1],sizeof(PetscReal)*nqp*nen*dim);CHKERRQ(ierr);
    ierr = PetscMemzero(point->shape[2],sizeof(PetscReal)*nqp*nen*dim*dim);CHKERRQ(ierr);
    ierr = PetscMemzero(point->shape[3],sizeof(PetscReal)*nqp*nen*dim*dim*dim);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "IGAColPointBegin"
PetscErrorCode IGAColPointBegin(IGAColPoint point)
{
  IGA            iga;
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  iga = point->parent;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,0);
  if (PetscUnlikely(!iga->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call IGASetUp() first");
  point->index = -1;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "_FindSpan"
PetscInt _FindSpan(PetscInt n,PetscInt p,PetscReal u, PetscReal *U)
{
  /* n is the index of the last basis */
  PetscInt low,high,span;
  if(u >= U[n+1]) return n;
  if(u <= U[p]) return p;
  low  = p;
  high = n+1;
  span = (high+low)/2;
  while(u < U[span] || u >= U[span+1]){
    if(u < U[span])
      high = span;
    else
      low = span;
    span = (high+low)/2;
  }
  return span;
}

#undef  __FUNCT__
#define __FUNCT__ "_Greville"
PetscReal _Greville(PetscInt i,PetscInt p,PetscInt m,PetscReal *U)
{
  PetscInt j;
  PetscReal X = 0.0;
  for(j=0;j<p;j++) X += U[i+j+1];
  X /= p;
  return X;
}

#undef  __FUNCT__
#define __FUNCT__ "IGAColPointNext"
PetscBool IGAColPointNext(IGAColPoint point)
{
  IGA      iga = point->parent;
  PetscInt i,dim  = point->dim;
  PetscInt *start = point->start;
  PetscInt *width = point->width;
  PetscInt *ID    = point->ID;
  PetscInt *span  = point->span;
  PetscReal *pnt  = point->point;
  PetscInt index,coord;
 
  point->nvec = 0;
  point->nmat = 0;

  index = ++point->index;
  if (PetscUnlikely(index >= point->count)) {
    point->index = -1;
    return PETSC_FALSE;
  }
  /* */
  for (i=0; i<dim; i++) {
    coord   = index % width[i];
    index   = (index - coord) / width[i];
    ID[i]   = coord + start[i]; 
    pnt[i]  = _Greville(ID[i],iga->axis[i]->p,iga->axis[i]->m,iga->axis[i]->U);
    span[i] = _FindSpan(iga->axis[i]->m-iga->axis[i]->p-1,iga->axis[i]->p,pnt[i],iga->axis[i]->U);
  }
  for (i=dim; i<3; i++) {
    span[i] = 0;
  }
  //printf("ID: {%d,%d,%d}  pnt: {%.2f,%.2f,%.2f}  span: {%d,%d,%d}\n",ID[0],ID[1],ID[2],pnt[0],pnt[1],pnt[2],span[0],span[1],span[2]);
  IGAColPointBuildMapping(point);
  IGAColPointBuildGeometry(point);
  IGAColPointBuildShapeFuns(point);
  return PETSC_TRUE;
}

#undef  __FUNCT__
#define __FUNCT__ "IGAColPointEnd"
PetscErrorCode IGAColPointEnd(IGAColPoint point)
{
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  point->index = -1;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAColPointBuildMapping"
PetscErrorCode IGAColPointBuildMapping(IGAColPoint point)
{
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  if (PetscUnlikely(point->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during collocation point loop");
  { /* */
    IGA      iga = point->parent;
    IGABasis *BD = iga->basis;
    PetscInt *span = point->span;
    PetscInt ia, inen = BD[0]->nen, ioffset = span[0]-iga->axis[0]->p;
    PetscInt ja, jnen = BD[1]->nen, joffset = span[1]-iga->axis[1]->p;
    PetscInt ka, knen = BD[2]->nen, koffset = span[2]-iga->axis[2]->p;
    PetscInt *start = iga->node_gstart, *width = iga->node_gwidth;
    PetscInt istart = start[0]/*istride = 1*/;
    PetscInt jstart = start[1], jstride = width[0];
    PetscInt kstart = start[2], kstride = width[0]*width[1];
    PetscInt a=0, *mapping = point->mapping;
    for (ka=0; ka<knen; ka++) {
      for (ja=0; ja<jnen; ja++) {
        for (ia=0; ia<inen; ia++) {
          PetscInt iA = (ioffset + ia) - istart;
          PetscInt jA = (joffset + ja) - jstart;
          PetscInt kA = (koffset + ka) - kstart;
          mapping[a++] = iA + jA*jstride + kA*kstride;
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAColPointBuildGeometry"
PetscErrorCode IGAColPointBuildGeometry(IGAColPoint point)
{
  IGA iga;
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  if (PetscUnlikely(point->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during collocation point loop");
  iga = point->parent;
  point->geometry = iga->geometry;
  point->rational = iga->rational;
  if (point->geometry || point->rational) {
    const PetscInt  *map = point->mapping;
    const PetscReal *arrayX = iga->geometryX;
    const PetscReal *arrayW = iga->geometryW;
    PetscReal *X = point->geometryX;
    PetscReal *W = point->geometryW;
    PetscInt a,nen = point->nen;
    PetscInt i,nsd = point->nsd;
    if (point->geometry)
      for (a=0; a<nen; a++)
        for (i=0; i<nsd; i++)
          X[i + a*nsd] = arrayX[i + map[a]*nsd];
    if (point->rational)
      for (a=0; a<nen; a++)
        W[a] = arrayW[map[a]];
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
extern void IGA_BasisFuns_1D(PetscInt,PetscInt,const PetscReal[],
                             PetscInt,PetscInt,PetscInt,const PetscReal[],
                             PetscReal[],PetscReal[],PetscReal[],PetscReal[]);
extern void IGA_BasisFuns_2D(PetscInt,PetscInt,const PetscReal[],
                             PetscInt,PetscInt,PetscInt,const PetscReal[],
                             PetscInt,PetscInt,PetscInt,const PetscReal[],
                             PetscReal[],PetscReal[],PetscReal[],PetscReal[]);
extern void IGA_BasisFuns_3D(PetscInt,PetscInt,const PetscReal[],
                             PetscInt,PetscInt,PetscInt,const PetscReal[],
                             PetscInt,PetscInt,PetscInt,const PetscReal[],
                             PetscInt,PetscInt,PetscInt,const PetscReal[],
                             PetscReal[],PetscReal[],PetscReal[],PetscReal[]);
EXTERN_C_END

EXTERN_C_BEGIN
extern void IGA_ShapeFuns_1D(PetscInt,PetscInt,PetscInt,const PetscReal[],
                             const PetscReal[],const PetscReal[],const PetscReal[],const PetscReal[],
                             PetscReal[],PetscReal[],PetscReal[],PetscReal[],
                             PetscReal[],PetscReal[],PetscReal[]);
extern void IGA_ShapeFuns_2D(PetscInt,PetscInt,PetscInt,const PetscReal[],
                             const PetscReal[],const PetscReal[],const PetscReal[],const PetscReal[],
                             PetscReal[],PetscReal[],PetscReal[],PetscReal[],
                             PetscReal[],PetscReal[],PetscReal[]);
extern void IGA_ShapeFuns_3D(PetscInt,PetscInt,PetscInt,const PetscReal[],
                             const PetscReal[],const PetscReal[],const PetscReal[],const PetscReal[],
                             PetscReal[],PetscReal[],PetscReal[],PetscReal[],
                             PetscReal[],PetscReal[],PetscReal[]);
EXTERN_C_END

EXTERN_C_BEGIN
extern void IGA_DersBasisFuns(PetscInt i,PetscReal u,PetscInt p,PetscInt d,const PetscReal U[],PetscReal N[]);
EXTERN_C_END

#define IGA_BasisFuns_ARGS(BD,i) 1,BD[i]->nen,BD[i]->d,point->basis1d[i]

#undef  __FUNCT__
#define __FUNCT__ "IGAColPointBuildShapeFuns"
PetscErrorCode IGAColPointBuildShapeFuns(IGAColPoint point)
{
  PetscInt order;
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  if (PetscUnlikely(point->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during collocation point loop");
  order = point->parent->order;
  {
    IGABasis *BD  = point->parent->basis;
    PetscReal **N = point->basis;
    switch (point->dim) {
    case 3: 
      IGA_DersBasisFuns(point->span[0],point->point[0],point->parent->axis[0]->p,point->parent->order,point->parent->axis[0]->U,point->basis1d[0]);
      IGA_DersBasisFuns(point->span[1],point->point[1],point->parent->axis[1]->p,point->parent->order,point->parent->axis[1]->U,point->basis1d[1]);
      IGA_DersBasisFuns(point->span[2],point->point[2],point->parent->axis[2]->p,point->parent->order,point->parent->axis[2]->U,point->basis1d[2]);
      IGA_BasisFuns_3D(order,point->rational,
		       point->geometryW,
		       IGA_BasisFuns_ARGS(BD,0),
		       IGA_BasisFuns_ARGS(BD,1),
		       IGA_BasisFuns_ARGS(BD,2),
		       N[0],N[1],N[2],N[3]); break;
    case 2: 
      IGA_DersBasisFuns(point->span[0],point->point[0],point->parent->axis[0]->p,point->parent->order,point->parent->axis[0]->U,point->basis1d[0]);
      IGA_DersBasisFuns(point->span[1],point->point[1],point->parent->axis[1]->p,point->parent->order,point->parent->axis[1]->U,point->basis1d[1]);
      IGA_BasisFuns_2D(order,point->rational,
		       point->geometryW,
		       IGA_BasisFuns_ARGS(BD,0),
		       IGA_BasisFuns_ARGS(BD,1),
		       N[0],N[1],N[2],N[3]); break;
    case 1: 
      IGA_DersBasisFuns(point->span[0],point->point[0],point->parent->axis[0]->p,point->parent->order,point->parent->axis[0]->U,point->basis1d[0]);
      IGA_BasisFuns_1D(order,point->rational,
		       point->geometryW,
		       IGA_BasisFuns_ARGS(BD,0),
		       N[0],N[1],N[2],N[3]); break;
    }
  }
  if (point->dim == point->nsd) /* XXX */
  if (point->geometry) {
    PetscReal **M = point->basis;
    PetscReal **N = point->shape;
    PetscReal dX  = point->detJac;
    PetscReal **gX = point->gradX;
    switch (point->dim) {
    case 3: IGA_ShapeFuns_3D(order,
                             1,point->nen,
                             point->geometryX,
                             M[0],M[1],M[2],M[3],
                             N[0],N[1],N[2],N[3],
                             &dX,gX[0],gX[1]); break;
    case 2: IGA_ShapeFuns_2D(order,
                             1,point->nen,
                             point->geometryX,
                             M[0],M[1],M[2],M[3],
                             N[0],N[1],N[2],N[3],
                             &dX,gX[0],gX[1]); break;
    case 1: IGA_ShapeFuns_1D(order,
                             1,point->nen,
                             point->geometryX,
                             M[0],M[1],M[2],M[3],
                             N[0],N[1],N[2],N[3],
                             &dX,gX[0],gX[1]); break;
    }
  }
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "IGAColPointGetIndex"
PetscErrorCode IGAColPointGetIndex(IGAColPoint point,PetscInt *index)
{
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Function not implemented");

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAColPointGetSizes"
PetscErrorCode IGAColPointGetSizes(IGAColPoint point,PetscInt *nen,PetscInt *dof,PetscInt *dim)
{
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  if (nen) PetscValidIntPointer(nen,2);
  if (dof) PetscValidIntPointer(dof,3);
  if (dim) PetscValidIntPointer(dim,4);
  if (nen) *nen = point->nen;
  if (dof) *dof = point->dof;
  if (dim) *dim = point->dim;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAColPointGetMapping"
PetscErrorCode IGAColPointGetMapping(IGAColPoint point,PetscInt *nen,const PetscInt *mapping[])
{
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Function not implemented");

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAColPointGetShapeFuns"
PetscErrorCode IGAColPointGetShapeFuns(IGAColPoint point,PetscInt der,const PetscReal *shapefuns[])
{
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  PetscValidPointer(shapefuns,2);
  if (PetscUnlikely(der < 0 || der >= (PetscInt)(sizeof(point->shape)/sizeof(PetscReal*))))
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
            "Requested derivative must be in range [0,%d], got %D",
            (int)(sizeof(point->shape)/sizeof(PetscReal*)-1),der);
  *shapefuns = point->shape[der];
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAColPointGetBasisFuns"
PetscErrorCode IGAColPointGetBasisFuns(IGAColPoint point,PetscInt der,const PetscReal *basisfuns[])
{
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  PetscValidPointer(basisfuns,2);
  if (PetscUnlikely(der < 0 || der >= (PetscInt)(sizeof(point->basis)/sizeof(PetscReal*))))
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
            "Requested derivative must be in range [0,%d], got %D",
            (int)(sizeof(point->basis)/sizeof(PetscReal*)-1),der);
  *basisfuns = point->basis[der];
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAColPointGetWorkVec"
PetscErrorCode IGAColPointGetWorkVec(IGAColPoint point,PetscScalar *V[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  PetscValidPointer(V,2);
  if (PetscUnlikely(point->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during colocation point loop");
  {
    size_t MAX_WORK_VEC = sizeof(point->wvec)/sizeof(PetscScalar*);
    PetscInt n = point->dof;
    if (PetscUnlikely(point->nvec >= (PetscInt)MAX_WORK_VEC))
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many work vectors requested");
    if (PetscUnlikely(!point->wvec[point->nvec])) {
      ierr = PetscMalloc1(n,PetscScalar,&point->wvec[point->nvec]);CHKERRQ(ierr);
    }
    *V = point->wvec[point->nvec++];
    ierr = PetscMemzero(*V,n*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAColPointGetWorkMat"
PetscErrorCode IGAColPointGetWorkMat(IGAColPoint point,PetscScalar *M[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  PetscValidPointer(M,2);
  if (PetscUnlikely(point->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during colocation point loop");
  {
    size_t MAX_WORK_MAT = sizeof(point->wmat)/sizeof(PetscScalar*);
    PetscInt n = point->nen * point->dof;
    if (PetscUnlikely(point->nmat >= (PetscInt)MAX_WORK_MAT))
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many work matrices requested");
    if (PetscUnlikely(!point->wmat[point->nmat])) {
      ierr = PetscMalloc1(n,PetscScalar,&point->wmat[point->nmat]);CHKERRQ(ierr);
    }
    *M = point->wmat[point->nmat++];
    ierr = PetscMemzero(*M,n*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAColPointGetValues"
PetscErrorCode IGAColPointGetValues(IGAColPoint point,const PetscScalar U[],PetscScalar u[])
{
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Function not implemented");
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "IGAColPointAssembleVec"
PetscErrorCode IGAColPointAssembleVec(IGAColPoint point,const PetscScalar F[],Vec vec)
{
  PetscInt       index;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  PetscValidScalarPointer(F,2);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,3);
  index = point->index;
  if (point->dof == 1) {
    ierr = VecSetValuesLocal(vec,1,&index,F,ADD_VALUES);CHKERRQ(ierr);
  } else {
    ierr = VecSetValuesBlockedLocal(vec,1,&index,F,ADD_VALUES);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAColPointAssembleMat"
PetscErrorCode IGAColPointAssembleMat(IGAColPoint point,const PetscScalar K[],Mat mat)
{
  PetscInt       nn,index,*ii;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(point,1);
  PetscValidScalarPointer(K,2);
  PetscValidHeaderSpecific(mat,MAT_CLASSID,3);
  nn = point->nen;
  ii = point->mapping;
  index = point->index;
  if (point->dof == 1) {
    ierr = MatSetValuesLocal(mat,1,&index,nn,ii,K,ADD_VALUES);CHKERRQ(ierr);
  } else {
    ierr = MatSetValuesBlockedLocal(mat,1,&index,nn,ii,K,ADD_VALUES);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAColComputeSystem"
PetscErrorCode IGAColComputeSystem(IGA iga,Mat matA,Vec vecB)
{
  IGAColUserSystem  System;
  void           *SysCtx;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(matA,MAT_CLASSID,2);
  PetscValidHeaderSpecific(vecB,VEC_CLASSID,3);
  IGACheckSetUp(iga,1);
  IGACheckUserOp(iga,1,ColSystem);
  System = iga->userops->ColSystem;
  SysCtx = iga->userops->ColSysCtx;
  ierr = IGAColFormSystem(iga,matA,vecB,System,SysCtx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAColFormSystem"
PetscErrorCode IGAColFormSystem(IGA iga,Mat matA,Vec vecB,IGAColUserSystem System,void *ctx)
{
  IGAColPoint     point;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(matA,MAT_CLASSID,2);
  PetscValidHeaderSpecific(vecB,VEC_CLASSID,3);
  IGACheckSetUp(iga,1);

  ierr = MatZeroEntries(matA);CHKERRQ(ierr);
  ierr = VecZeroEntries(vecB);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(IGA_ColFormSystem,iga,matA,vecB,0);CHKERRQ(ierr);
  ierr = IGAGetColPoint(iga,&point);CHKERRQ(ierr);
  ierr = IGAColPointBegin(point);CHKERRQ(ierr);
  while (IGAColPointNext(point)) {
    PetscScalar *K, *F;
    ierr = IGAColPointGetWorkMat(point,&K);CHKERRQ(ierr);
    ierr = IGAColPointGetWorkVec(point,&F);CHKERRQ(ierr);
    ierr = System(point,K,F,ctx);CHKERRQ(ierr);
    ierr = IGAColPointAssembleMat(point,K,matA);CHKERRQ(ierr);
    ierr = IGAColPointAssembleVec(point,F,vecB);CHKERRQ(ierr);
  }
  ierr = IGAColPointEnd(point);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(IGA_ColFormSystem,iga,matA,vecB,0);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(matA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (matA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(vecB);CHKERRQ(ierr);
  ierr = VecAssemblyEnd  (vecB);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAColSetUserSystem"
PetscErrorCode IGAColSetUserSystem(IGA iga,IGAColUserSystem System,void *SysCtx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  if (System) iga->userops->ColSystem = System;
  if (SysCtx) iga->userops->ColSysCtx = SysCtx;
  PetscFunctionReturn(0);
}
