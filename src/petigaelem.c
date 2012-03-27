#include "petiga.h"

#undef  __FUNCT__
#define __FUNCT__ "IGAElementCreate"
PetscErrorCode IGAElementCreate(IGAElement *_element)
{
  IGAElement element;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(_element,1);
  ierr = PetscNew(struct _n_IGAElement,_element);CHKERRQ(ierr);
  element = *_element;
  element->refct =  1;
  element->index = -1;
  ierr = IGAPointCreate(&element->iterator);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAElementDestroy"
PetscErrorCode IGAElementDestroy(IGAElement *_element)
{
  IGAElement     element;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(_element,1);
  element = *_element; *_element = 0;
  if (!element) PetscFunctionReturn(0);
  if (--element->refct > 0) PetscFunctionReturn(0);
  ierr = IGAPointDestroy(&element->iterator);CHKERRQ(ierr);
  ierr = IGAElementReset(element);CHKERRQ(ierr);
  ierr = PetscFree(element);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAElementFreeWork"
static
PetscErrorCode IGAElementFreeWork(IGAElement element)
{
  size_t i;
  size_t MAX_WORK_VEC;
  size_t MAX_WORK_MAT;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  MAX_WORK_VEC = sizeof(element->wvec)/sizeof(PetscScalar*);
  for (i=0; i<MAX_WORK_VEC; i++)
    {ierr = PetscFree(element->wvec[i]);CHKERRQ(ierr);}
  element->nvec = 0;
  MAX_WORK_MAT = sizeof(element->wmat)/sizeof(PetscScalar*);
  for (i=0; i<MAX_WORK_MAT; i++)
    {ierr = PetscFree(element->wmat[i]);CHKERRQ(ierr);}
  element->nmat = 0;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAElementReset"
PetscErrorCode IGAElementReset(IGAElement element)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!element) PetscFunctionReturn(0);
  PetscValidPointer(element,1);
  element->index = -1;
  ierr = PetscFree(element->mapping);CHKERRQ(ierr);
  ierr = PetscFree(element->geometry);CHKERRQ(ierr);
  ierr = PetscFree(element->point);CHKERRQ(ierr);
  ierr = PetscFree(element->weight);CHKERRQ(ierr);
  ierr = PetscFree(element->detJ);CHKERRQ(ierr);
  ierr = PetscFree(element->jacobian);CHKERRQ(ierr);
  ierr = PetscFree(element->shape[0]);CHKERRQ(ierr);
  ierr = PetscFree(element->shape[1]);CHKERRQ(ierr);
  ierr = PetscFree(element->shape[2]);CHKERRQ(ierr);
  ierr = PetscFree(element->shape[3]);CHKERRQ(ierr);
  ierr = IGAElementFreeWork(element);CHKERRQ(ierr);
  ierr = PetscFree(element->ifix);CHKERRQ(ierr);
  ierr = PetscFree(element->vfix);CHKERRQ(ierr);
  ierr = PetscFree(element->xfix);CHKERRQ(ierr);
  ierr = IGAPointReset(element->iterator);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAElementSetUp"
PetscErrorCode IGAElementSetUp(IGAElement element)
{
  IGA            iga;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  iga = element->parent;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,0);
  if (PetscUnlikely(!iga->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call IGASetUp() first");
  ierr = IGAElementReset(element);CHKERRQ(ierr);

  element->dim = iga->dim;
  element->dof = iga->dof;
  { /* */
    IGABasis *BD = iga->basis;
    PetscInt i,dim = element->dim;
    PetscInt nel=1, nen=1, nqp=1;
    PetscInt start[3],width[3];
    ierr = DMDAGetGhostCorners(iga->dm_dof,
                               &start[0],&start[1],&start[2],
                               &width[0],&width[1],&width[2]);CHKERRQ(ierr);
    for (i=0; i<dim; i++) {
      element->range[i][0] = iga->elem_start[i];
      element->range[i][1] = iga->elem_start[i] + iga->elem_width[i];
      element->start[i]    = start[i];
      element->width[i]    = width[i];
      nel *= iga->elem_width[i];
      nen *= BD[i]->nen;
      nqp *= BD[i]->nqp;
    }
    for (i=dim; i<3; i++) {
      element->range[i][0] = 0;
      element->range[i][1] = 1;
      element->start[i] = 0;
      element->width[i] = 1;
    }
    element->index = -1;
    element->count = nel;
    element->nqp   = nqp;
    element->nen   = nen;
  }
  { /* */
    PetscInt dim = element->dim;
    PetscInt nqp = element->nqp;
    PetscInt nen = element->nen;

    ierr = PetscMalloc1(nen,PetscInt,&element->mapping);CHKERRQ(ierr);
    ierr = PetscMalloc1(nen*(dim+1),PetscReal,&element->geometry);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*dim,PetscReal,&element->point);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp,PetscReal,&element->weight);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp,PetscReal,&element->detJ);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*dim*dim,PetscReal,&element->jacobian);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*nen,PetscReal,&element->shape[0]);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*nen*dim,PetscReal,&element->shape[1]);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*nen*dim*dim,PetscReal,&element->shape[2]);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*nen*dim*dim*dim,PetscReal,&element->shape[3]);CHKERRQ(ierr);

    ierr = PetscMemzero(element->detJ,    sizeof(PetscReal)*nqp);CHKERRQ(ierr);
    ierr = PetscMemzero(element->jacobian,sizeof(PetscReal)*nqp*dim*dim);CHKERRQ(ierr);
    ierr = PetscMemzero(element->shape[0],sizeof(PetscReal)*nqp*nen);CHKERRQ(ierr);
    ierr = PetscMemzero(element->shape[1],sizeof(PetscReal)*nqp*nen*dim);CHKERRQ(ierr);
    ierr = PetscMemzero(element->shape[2],sizeof(PetscReal)*nqp*nen*dim*dim);CHKERRQ(ierr);
    ierr = PetscMemzero(element->shape[3],sizeof(PetscReal)*nqp*nen*dim*dim*dim);CHKERRQ(ierr);
  }
  element->iterator->parent = element;
  ierr = IGAPointSetUp(element->iterator);CHKERRQ(ierr);
  { /* */
    PetscInt nen = element->nen;
    PetscInt dof = element->dof;
    element->nfix = 0;
    ierr = PetscMalloc1(nen*dof,PetscInt,   &element->ifix);CHKERRQ(ierr);
    ierr = PetscMalloc1(nen*dof,PetscScalar,&element->vfix);CHKERRQ(ierr);
    ierr = PetscMalloc1(nen*dof,PetscScalar,&element->xfix);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAElementBegin"
PetscErrorCode IGAElementBegin(IGAElement element)
{
  IGA            iga;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  iga = element->parent;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,0);
  if (PetscUnlikely(!iga->setup)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call IGASetUp() first");
  if (iga->vec_geom) {
    ierr = VecGetArrayRead(iga->vec_geom,&iga->geometry);CHKERRQ(ierr);
  }
  element->index = -1;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAElementNext"
PetscBool IGAElementNext(IGAElement element)
{
  PetscInt i,dim = element->dim;
  PetscInt *ID   = element->ID;
  PetscInt index;
  /* */
  element->nvec = 0;
  element->nmat = 0;
  /* */
  index = ++element->index;
  if (PetscUnlikely(index >= element->count)) {
    element->index = -1;
    return PETSC_FALSE;
  }
  /* */
  for (i=0; i<dim; i++) {
    PetscInt *range = element->range[i];
    PetscInt size  = range[1] - range[0];
    PetscInt coord = index % size;
    index = (index - coord) / size;
    ID[i] = coord + range[0];
  }
  IGAElementBuildMap(element);
  IGAElementBuildFix(element);
  return PETSC_TRUE;
}

#undef  __FUNCT__
#define __FUNCT__ "IGAElementEnd"
PetscErrorCode IGAElementEnd(IGAElement element)
{
  IGA            iga;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  iga = element->parent;
  if (iga->vec_geom) {
    ierr = VecRestoreArrayRead(iga->vec_geom,&iga->geometry);CHKERRQ(ierr);
  }
  element->index = -1;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAElementGetParent"
PetscErrorCode IGAElementGetParent(IGAElement element,IGA *parent)
{
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  PetscValidIntPointer(parent,2);
  *parent = element->parent;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAElementGetIndex"
PetscErrorCode IGAElementGetIndex(IGAElement element,PetscInt *index)
{
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  PetscValidIntPointer(index,2);
  *index = element->index;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAElementGetInfo"
PetscErrorCode IGAElementGetInfo(IGAElement element,PetscInt *nqp, PetscInt *nen,PetscInt *dof,PetscInt *dim)
{
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  if (nqp) PetscValidIntPointer(nqp,4);
  if (nen) PetscValidIntPointer(nen,5);
  if (dof) PetscValidIntPointer(dof,3);
  if (dim) PetscValidIntPointer(dim,2);
  if (nqp) *nqp = element->nqp;
  if (nen) *nen = element->nen;
  if (dof) *dof = element->dof;
  if (dim) *dim = element->dim;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAElementGetPoint"
PetscErrorCode IGAElementGetPoint(IGAElement element,IGAPoint *point)
{
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  PetscValidPointer(point,2);
  *point = element->iterator;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

static void AddFix(IGAElement element,PetscInt dir,PetscInt side,PetscInt a)
{
  IGABoundary b = element->parent->boundary[dir][side];
  PetscInt dof = element->dof;
  PetscInt nfix = element->nfix;
  PetscInt *ifix = element->ifix;
  PetscScalar *vfix = element->vfix;
  PetscInt j,k,n = b->nbc;
  for (k=0; k<n; k++) {
    PetscInt idx = a*dof + b->field[k];
    PetscScalar val = b->value[k];
    for (j=0; j<nfix; j++)
      if (ifix[j] == idx) break;
    if (j==nfix) nfix++;
    ifix[j] = idx;
    vfix[j] = val;
  }
  element->nfix = nfix;
}

#undef  __FUNCT__
#define __FUNCT__ "IGAElementBuildFix"
PetscErrorCode IGAElementBuildFix(IGAElement element)
{
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  if (PetscUnlikely(element->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during element loop");
  {
    IGAAxis  *AX = element->parent->axis;
    IGABasis *BD = element->parent->basis;
    PetscInt *ID = element->ID;
    PetscInt A0[3] = {PETSC_MIN_INT,PETSC_MIN_INT,PETSC_MIN_INT};
    PetscInt A1[3] = {PETSC_MAX_INT,PETSC_MAX_INT,PETSC_MAX_INT};
    PetscBool onboundary = PETSC_FALSE;
    PetscInt i,dim = element->dim;
    for (i=0; i<dim; i++) {
      PetscInt e = BD[i]->nel-1;
      PetscInt n = BD[i]->nnp-1;
      if (AX[i]->periodic) continue;
      if (ID[i] == 0) { A0[i] = 0; onboundary = PETSC_TRUE; }
      if (ID[i] == e) { A1[i] = n; onboundary = PETSC_TRUE; }
    }
    element->nfix = 0;
    if (onboundary) {
      PetscInt ia, inen = BD[0]->nen, ioffset = BD[0]->offset[ID[0]];
      PetscInt ja, jnen = BD[1]->nen, joffset = BD[1]->offset[ID[1]];
      PetscInt ka, knen = BD[2]->nen, koffset = BD[2]->offset[ID[2]];
      PetscInt a = 0;
      for (ka=0; ka<knen; ka++) {
        for (ja=0; ja<jnen; ja++) {
          for (ia=0; ia<inen; ia++) {
            PetscInt iA = ioffset + ia;
            PetscInt jA = joffset + ja;
            PetscInt kA = koffset + ka;
            /**/ if (iA == A0[0]) AddFix(element,0,0,a);
            else if (iA == A1[0]) AddFix(element,0,1,a);
            /**/ if (jA == A0[1]) AddFix(element,1,0,a);
            else if (jA == A1[1]) AddFix(element,1,1,a);
            /**/ if (kA == A0[2]) AddFix(element,2,0,a);
            else if (kA == A1[2]) AddFix(element,2,1,a);
            a++;
          }
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAElementBuildMap"
PetscErrorCode IGAElementBuildMap(IGAElement element)
{
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  if (PetscUnlikely(element->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during element loop");
  { /* */
    IGABasis *BD = element->parent->basis;
    PetscInt *ID = element->ID;
    PetscInt *start = element->start;
    PetscInt *width = element->width;
    PetscInt ia, inen = BD[0]->nen, ioffset = BD[0]->offset[ID[0]];
    PetscInt ja, jnen = BD[1]->nen, joffset = BD[1]->offset[ID[1]];
    PetscInt ka, knen = BD[2]->nen, koffset = BD[2]->offset[ID[2]];
    PetscInt istart = start[0], istride = 1;
    PetscInt jstart = start[1], jstride = width[0];
    PetscInt kstart = start[2], kstride = width[0]*width[1];
    PetscInt a=0, *mapping = element->mapping;
    for (ka=0; ka<knen; ka++) {
      for (ja=0; ja<jnen; ja++) {
        for (ia=0; ia<inen; ia++) {
          PetscInt iA = (ioffset + ia) - istart;
          PetscInt jA = (joffset + ja) - jstart;
          PetscInt kA = (koffset + ka) - kstart;
          mapping[a++] = iA*istride + jA*jstride + kA*kstride;
        }
      }
    }
  }
  if (element->parent->geometry) {
    const PetscScalar *arrayX = element->parent->geometry;
    PetscReal *Cw = element->geometry;
    PetscInt *mapping = element->mapping;
    PetscInt nen = element->nen;
    PetscInt dim = element->dim;
    PetscInt a,i,pos=0;
    for (a=0; a<nen; a++) {
      const PetscScalar *X = arrayX + mapping[a]*(dim+1);
      for (i=0; i<(dim+1); i++)
        Cw[pos++] = PetscRealPart(X[i]);
    }
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
extern void IGA_Quadrature_1D(PetscInt,const PetscReal[],const PetscReal[],
                              PetscReal[],PetscReal[]);
extern void IGA_Quadrature_2D(PetscInt,const PetscReal[],const PetscReal[],
                              PetscInt,const PetscReal[],const PetscReal[],
                              PetscReal[],PetscReal[]);
extern void IGA_Quadrature_3D(PetscInt,const PetscReal[],const PetscReal[],
                              PetscInt,const PetscReal[],const PetscReal[],
                              PetscInt,const PetscReal[],const PetscReal[],
                              PetscReal[],PetscReal[]);
EXTERN_C_END

#define IGA_Quadrature_ARGS(ID,BD,i) \
  BD[i]->nqp,BD[i]->point+ID[i]*BD[i]->nqp,BD[i]->weight

#undef  __FUNCT__
#define __FUNCT__ "IGAElementBuildQuadrature"
PetscErrorCode IGAElementBuildQuadrature(IGAElement element)
{
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  if (PetscUnlikely(element->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during element loop");
  {
    IGABasis *BD = element->parent->basis;
    PetscInt *ID = element->ID;
    switch (element->dim) {
    case 3: IGA_Quadrature_3D(IGA_Quadrature_ARGS(ID,BD,0),
                              IGA_Quadrature_ARGS(ID,BD,1),
                              IGA_Quadrature_ARGS(ID,BD,2),
                              element->point,element->weight); break;
    case 2: IGA_Quadrature_2D(IGA_Quadrature_ARGS(ID,BD,0),
                              IGA_Quadrature_ARGS(ID,BD,1),
                              element->point,element->weight); break;
    case 1: IGA_Quadrature_1D(IGA_Quadrature_ARGS(ID,BD,0),
                              element->point,element->weight); break;
    }
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
extern void IGA_ShapeFuns_1D(PetscInt,PetscInt,
                             PetscInt,PetscInt,PetscInt,const PetscReal[],const PetscReal[],
                             const PetscReal[],PetscReal[],PetscReal[],PetscReal[],PetscReal[],PetscReal[],PetscReal[]);
extern void IGA_ShapeFuns_2D(PetscInt,PetscInt,
                             PetscInt,PetscInt,PetscInt,const PetscReal[],const PetscReal[],
                             PetscInt,PetscInt,PetscInt,const PetscReal[],const PetscReal[],
                             const PetscReal[],PetscReal[],PetscReal[],PetscReal[],PetscReal[],PetscReal[],PetscReal[]);
extern void IGA_ShapeFuns_3D(PetscInt,PetscInt,
                             PetscInt,PetscInt,PetscInt,const PetscReal[],const PetscReal[],
                             PetscInt,PetscInt,PetscInt,const PetscReal[],const PetscReal[],
                             PetscInt,PetscInt,PetscInt,const PetscReal[],const PetscReal[],
                             const PetscReal[],PetscReal[],PetscReal[],PetscReal[],PetscReal[],PetscReal[],PetscReal[]);
EXTERN_C_END

#define IGA_ShapeFuns_ARGS(ID,BD,i) \
  BD[i]->nqp,BD[i]->nen,BD[i]->d,   \
  BD[i]->detJ+ID[i],BD[i]->value+ID[i]*BD[i]->nqp*BD[i]->nen*(BD[i]->d+1)

#undef  __FUNCT__
#define __FUNCT__ "IGAElementBuildShapeFuns"
PetscErrorCode IGAElementBuildShapeFuns(IGAElement element)
{
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  if (PetscUnlikely(element->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during element loop");
  {
    PetscBool geometry = element->parent->geometry ? PETSC_TRUE : PETSC_FALSE;
    PetscBool rational = geometry ? element->parent->rational : PETSC_FALSE;
    IGABasis *BD  = element->parent->basis;
    PetscInt *ID  = element->ID;
    PetscReal *Cw = element->geometry;
    PetscReal *dJ = element->detJ;
    PetscReal *J  = element->jacobian;
    PetscReal **N = element->shape;
    switch (element->dim) {
    case 3: IGA_ShapeFuns_3D(geometry,rational,
                             IGA_ShapeFuns_ARGS(ID,BD,0),
                             IGA_ShapeFuns_ARGS(ID,BD,1),
                             IGA_ShapeFuns_ARGS(ID,BD,2),
                             Cw,dJ,J,N[0],N[1],N[2],N[3]); break;
    case 2: IGA_ShapeFuns_2D(geometry,rational,
                             IGA_ShapeFuns_ARGS(ID,BD,0),
                             IGA_ShapeFuns_ARGS(ID,BD,1),
                             Cw,dJ,J,N[0],N[1],N[2],N[3]); break;
    case 1: IGA_ShapeFuns_1D(geometry,rational,
                             IGA_ShapeFuns_ARGS(ID,BD,0),
                             Cw,dJ,J,N[0],N[1],N[2],N[3]); break;
    }
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

#undef  __FUNCT__
#define __FUNCT__ "IGAElementGetWorkVec"
PetscErrorCode IGAElementGetWorkVec(IGAElement element,PetscScalar *V[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  PetscValidPointer(V,2);
  if (PetscUnlikely(element->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during element loop");
  {
    size_t MAX_WORK_VEC = sizeof(element->wvec)/sizeof(PetscScalar*);
    PetscInt n = element->nen * element->dof;
    if (PetscUnlikely(element->nvec >= (PetscInt)MAX_WORK_VEC))
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many work vectors requested");
    if (PetscUnlikely(!element->wvec[element->nvec])) {
      ierr = PetscMalloc1(n,PetscScalar,&element->wvec[element->nvec]);CHKERRQ(ierr);
    }
    *V = element->wvec[element->nvec++];
    ierr = PetscMemzero(*V,n*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAElementGetWorkMat"
PetscErrorCode IGAElementGetWorkMat(IGAElement element,PetscScalar *M[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  PetscValidPointer(M,2);
  if (PetscUnlikely(element->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during element loop");
  {
    size_t MAX_WORK_MAT = sizeof(element->wmat)/sizeof(PetscScalar*);
    PetscInt n = element->nen * element->dof;
    if (PetscUnlikely(element->nmat >= (PetscInt)MAX_WORK_MAT))
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many work matrices requested");
    if (PetscUnlikely(!element->wmat[element->nmat])) {
      ierr = PetscMalloc1(n*n,PetscScalar,&element->wmat[element->nmat]);CHKERRQ(ierr);
    }
    *M = element->wmat[element->nmat++];
    ierr = PetscMemzero(*M,n*n*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

#undef  __FUNCT__
#define __FUNCT__ "IGAElementGetValues"
PetscErrorCode IGAElementGetValues(IGAElement element,const PetscScalar arrayU[], PetscScalar U[])
{
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  PetscValidScalarPointer(arrayU,2);
  PetscValidScalarPointer(U,3);
  {
    PetscInt nen = element->nen;
    PetscInt dof = element->dof;
    PetscInt *mapping = element->mapping;
    PetscInt a,i,pos=0;
    for (a=0; a<nen; a++) {
      const PetscScalar *u = arrayU + mapping[a]*dof;
      for (i=0; i<dof; i++)
        U[pos++] = u[i]; /* XXX Use PetscMemcpy() ?? */
    }
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAElementFixValues"
PetscErrorCode IGAElementFixValues(IGAElement element,PetscScalar U[])
{
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  PetscValidScalarPointer(U,2);
  {
    PetscInt k,nfix=element->nfix;
    for (k=0; k<nfix; k++) {
      PetscInt i = element->ifix[k];
      element->xfix[k] = U[i];
      U[i] = element->vfix[k];
    }
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAElementFixFunction"
PetscErrorCode IGAElementFixFunction(IGAElement element,PetscScalar F[])
{
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  PetscValidScalarPointer(F,2);
  {
    PetscInt k,nfix=element->nfix;
    for (k=0; k<nfix; k++) {
      PetscInt i = element->ifix[k];
      PetscScalar u = element->xfix[k];
      PetscScalar u0 = element->vfix[k];
      F[i] = u - u0;
    }
  }
  PetscFunctionReturn(0);
}
#undef  __FUNCT__
#define __FUNCT__ "IGAElementFixJacobian"
PetscErrorCode IGAElementFixJacobian(IGAElement element,PetscScalar J[])
{
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  PetscValidScalarPointer(J,2);
  {
    PetscInt N = element->nen*element->dof;
    PetscInt k,nfix=element->nfix;
    for (k=0; k<nfix; k++) {
      PetscInt j,i = element->ifix[k];
      for (j=0; j<N; j++)
        J[i*N+j] = J[j*N+i] = 0.0;
      J[i*N+i] = 1.0;
    }
  }
  PetscFunctionReturn(0);
}
#undef  __FUNCT__
#define __FUNCT__ "IGAElementFixSystem"
PetscErrorCode IGAElementFixSystem(IGAElement element,PetscScalar K[],PetscScalar F[])
{
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  PetscValidScalarPointer(K,2);
  PetscValidScalarPointer(F,3);
  {
    PetscInt N = element->nen*element->dof;
    PetscInt k,nfix=element->nfix;
    for (k=0; k<nfix; k++) {
      PetscInt j,i = element->ifix[k];
      PetscScalar v = element->vfix[k];
      for (j=0; j<N; j++) {
        F[j] -= K[j*N+i] * v;
      }
      for (j=0; j<N; j++)
        K[i*N+j] = K[j*N+i] = 0.0;
      K[i*N+i] = 1.0;
      F[i] = v;
    }
  }
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "IGAElementAssembleVec"
PetscErrorCode IGAElementAssembleVec(IGAElement element,PetscScalar F[],Vec vec)
{
  PetscInt       nn,*ii;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  PetscValidScalarPointer(F,2);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,3);
  nn = element->nen;
  ii = element->mapping;
  if (element->dof == 1) {
    ierr = VecSetValuesLocal(vec,nn,ii,F,ADD_VALUES);CHKERRQ(ierr);
  } else {
    ierr = VecSetValuesBlockedLocal(vec,nn,ii,F,ADD_VALUES);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAElementAssembleMat"
PetscErrorCode IGAElementAssembleMat(IGAElement element,PetscScalar K[],Mat mat)
{
  PetscInt       nn,*ii;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  PetscValidScalarPointer(K,2);
  PetscValidHeaderSpecific(mat,MAT_CLASSID,3);
  nn = element->nen;
  ii = element->mapping;
  if (element->dof == 1) {
    ierr = MatSetValuesLocal(mat,nn,ii,nn,ii,K,ADD_VALUES);CHKERRQ(ierr);
  } else {
    ierr = MatSetValuesBlockedLocal(mat,nn,ii,nn,ii,K,ADD_VALUES);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
