#include "petiga.h"

PetscErrorCode IGAElementCreate(IGAElement *_element)
{
  IGAElement     element;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(_element,1);
  ierr = PetscCalloc1(1,&element);CHKERRQ(ierr);
  *_element = element; element->refct = 1;
  element->index = -1;
  ierr = IGAPointCreate(&element->iterator);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGAElementDestroy(IGAElement *_element)
{
  IGAElement     element;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(_element,1);
  element = *_element; *_element = NULL;
  if (!element) PetscFunctionReturn(0);
  if (--element->refct > 0) PetscFunctionReturn(0);
  ierr = IGAPointDestroy(&element->iterator);CHKERRQ(ierr);
  ierr = IGAElementReset(element);CHKERRQ(ierr);
  ierr = PetscFree(element);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGAElementReference(IGAElement element)
{
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  element->refct++;
  PetscFunctionReturn(0);
}

static
PetscErrorCode IGAElementFreeWork(IGAElement element)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  {
    size_t MAX_WORK_VAL = sizeof(element->wval)/sizeof(PetscScalar*);
    size_t MAX_WORK_VEC = sizeof(element->wvec)/sizeof(PetscScalar*);
    size_t MAX_WORK_MAT = sizeof(element->wmat)/sizeof(PetscScalar*);
    size_t i;
    for (i=0; i<MAX_WORK_VAL; i++)
      {ierr = PetscFree(element->wval[i]);CHKERRQ(ierr);}
    element->nval = 0;
    for (i=0; i<MAX_WORK_VEC; i++)
      {ierr = PetscFree(element->wvec[i]);CHKERRQ(ierr);}
    element->nvec = 0;
    for (i=0; i<MAX_WORK_MAT; i++)
      {ierr = PetscFree(element->wmat[i]);CHKERRQ(ierr);}
    element->nmat = 0;
  }
  PetscFunctionReturn(0);
}

static
PetscErrorCode IGAElementFreeFix(IGAElement element)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  element->nfix = 0;
  ierr = PetscFree(element->ifix);CHKERRQ(ierr);
  ierr = PetscFree(element->vfix);CHKERRQ(ierr);
  ierr = PetscFree(element->ufix);CHKERRQ(ierr);
  element->nflux = 0;
  ierr = PetscFree(element->iflux);CHKERRQ(ierr);
  ierr = PetscFree(element->vflux);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGAElementReset(IGAElement element)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!element) PetscFunctionReturn(0);
  PetscValidPointer(element,1);

  element->count =  0;
  element->index = -1;

  if (element->rowmap != element->mapping)
    {ierr = PetscFree(element->rowmap);CHKERRQ(ierr);}
  if (element->colmap != element->mapping)
    {ierr = PetscFree(element->colmap);CHKERRQ(ierr);}
  element->rowmap = element->colmap = NULL;
  ierr = PetscFree(element->mapping);CHKERRQ(ierr);
  ierr = PetscFree(element->rationalW);CHKERRQ(ierr);
  ierr = PetscFree(element->geometryX);CHKERRQ(ierr);
  ierr = PetscFree(element->propertyA);CHKERRQ(ierr);

  ierr = PetscFree(element->weight);CHKERRQ(ierr);
  ierr = PetscFree(element->detJac);CHKERRQ(ierr);

  ierr = PetscFree(element->basis[0]);CHKERRQ(ierr);
  ierr = PetscFree(element->basis[1]);CHKERRQ(ierr);
  ierr = PetscFree(element->basis[2]);CHKERRQ(ierr);
  ierr = PetscFree(element->basis[3]);CHKERRQ(ierr);
  ierr = PetscFree(element->basis[4]);CHKERRQ(ierr);

  ierr = PetscFree(element->shape[0]);CHKERRQ(ierr);
  ierr = PetscFree(element->shape[1]);CHKERRQ(ierr);
  ierr = PetscFree(element->shape[2]);CHKERRQ(ierr);
  ierr = PetscFree(element->shape[3]);CHKERRQ(ierr);
  ierr = PetscFree(element->shape[4]);CHKERRQ(ierr);

  ierr = PetscFree(element->mapU[0]);CHKERRQ(ierr);
  ierr = PetscFree(element->mapU[1]);CHKERRQ(ierr);
  ierr = PetscFree(element->mapU[2]);CHKERRQ(ierr);
  ierr = PetscFree(element->mapU[3]);CHKERRQ(ierr);
  ierr = PetscFree(element->mapU[4]);CHKERRQ(ierr);

  ierr = PetscFree(element->mapX[0]);CHKERRQ(ierr);
  ierr = PetscFree(element->mapX[1]);CHKERRQ(ierr);
  ierr = PetscFree(element->mapX[2]);CHKERRQ(ierr);
  ierr = PetscFree(element->mapX[3]);CHKERRQ(ierr);
  ierr = PetscFree(element->mapX[4]);CHKERRQ(ierr);

  ierr = PetscFree(element->normal);CHKERRQ(ierr);
  ierr = PetscFree(element->detX);CHKERRQ(ierr);
  ierr = PetscFree(element->detS);CHKERRQ(ierr);

  ierr = IGAElementFreeFix(element);CHKERRQ(ierr);
  ierr = IGAElementFreeWork(element);CHKERRQ(ierr);

  ierr = IGAPointReset(element->iterator);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode IGAElementInit(IGAElement element,IGA iga)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  PetscValidHeaderSpecific(iga,IGA_CLASSID,2);
  IGACheckSetUp(iga,1);

  ierr = IGAElementReset(element);CHKERRQ(ierr);
  element->parent = iga;
  element->collocation = iga->collocation;

  element->dof = iga->dof;
  element->dim = iga->dim;
  element->nsd = iga->geometry ? iga->geometry : iga->dim;
  element->npd = iga->property ? iga->property : 0;

  { /* */
    PetscInt *start = iga->elem_start;
    PetscInt *width = iga->elem_width;
    PetscInt *sizes = iga->elem_sizes;
    IGABasis *BD    = iga->basis;
    PetscInt i,dim = element->dim;
    PetscInt nel=1,nen=1,nqp=1;
    for (i=0; i<3; i++)
      element->ID[i] = 0;
    for (i=0; i<dim; i++) {
      element->start[i] = start[i];
      element->width[i] = width[i];
      element->sizes[i] = sizes[i];
      nel *= width[i];
      nen *= BD[i]->nen;
      nqp *= BD[i]->nqp;
    }
    for (i=dim; i<3; i++) {
      element->start[i] = 0;
      element->width[i] = 1;
      element->sizes[i] = 1;
    }
    element->index = -1;
    element->count = nel;
    element->nen   = nen;
    element->nqp   = nqp;
  }
  { /* */
    size_t nen = (size_t)element->nen;
    size_t nsd = (size_t)element->nsd;
    size_t npd = (size_t)element->npd;
    ierr = PetscMalloc1(nen,&element->mapping);CHKERRQ(ierr);
    if (PetscLikely(!element->collocation)) {
      element->neq = element->nen;
      element->rowmap = element->mapping;
    } else {
      element->neq = 1;
      ierr = PetscMalloc1(1,&element->rowmap);CHKERRQ(ierr);
    }
    element->colmap = element->mapping;
    ierr = PetscMalloc1(nen,&element->rationalW);CHKERRQ(ierr);
    ierr = PetscMalloc1(nen*nsd,&element->geometryX);CHKERRQ(ierr);
    ierr = PetscMalloc1(nen*npd,&element->propertyA);CHKERRQ(ierr);
  }
  { /* */
    size_t nqp = (size_t)element->nqp;
    size_t nen = (size_t)element->nen;
    size_t dim = (size_t)element->dim;
    size_t nsd = (size_t)element->nsd;

    ierr = PetscMalloc1(nqp,&element->weight);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp,&element->detJac);CHKERRQ(ierr);

    ierr = PetscMalloc1(nqp*nen,&element->basis[0]);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*nen*dim,&element->basis[1]);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*nen*dim*dim,&element->basis[2]);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*nen*dim*dim*dim,&element->basis[3]);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*nen*dim*dim*dim*dim,&element->basis[4]);CHKERRQ(ierr);

    ierr = PetscMalloc1(nqp*nen,&element->shape[0]);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*nen*nsd,&element->shape[1]);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*nen*nsd*nsd,&element->shape[2]);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*nen*nsd*nsd*nsd,&element->shape[3]);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*nen*nsd*nsd*nsd*nsd,&element->shape[4]);CHKERRQ(ierr);

    ierr = PetscMalloc1(nqp*dim,&element->mapU[0]);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*dim*nsd,&element->mapU[1]);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*dim*nsd*nsd,&element->mapU[2]);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*dim*nsd*nsd*nsd,&element->mapU[3]);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*dim*nsd*nsd*nsd*nsd,&element->mapU[4]);CHKERRQ(ierr);

    ierr = PetscMalloc1(nqp*nsd,&element->mapX[0]);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*nsd*dim,&element->mapX[1]);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*nsd*dim*dim,&element->mapX[2]);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*nsd*dim*dim*dim,&element->mapX[3]);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*nsd*dim*dim*dim*dim,&element->mapX[4]);CHKERRQ(ierr);

    ierr = PetscMalloc1(nqp*nsd,&element->normal);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp,&element->detX);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp,&element->detS);CHKERRQ(ierr);
  }
  { /* */
    size_t MAX_WORK_VAL = sizeof(element->wval)/sizeof(PetscScalar*);
    size_t MAX_WORK_VEC = sizeof(element->wvec)/sizeof(PetscScalar*);
    size_t MAX_WORK_MAT = sizeof(element->wmat)/sizeof(PetscScalar*);
    size_t i, n = (size_t)element->nen * (size_t)element->dof;
    for (i=0; i<MAX_WORK_VAL; i++)
      {ierr = PetscMalloc1(n,&element->wval[i]);CHKERRQ(ierr);}
    for (i=0; i<MAX_WORK_VEC; i++)
      {ierr = PetscMalloc1(n,&element->wvec[i]);CHKERRQ(ierr);}
    for (i=0; i<MAX_WORK_MAT; i++)
      {ierr = PetscMalloc1(n*n,&element->wmat[i]);CHKERRQ(ierr);}
  }
  { /* */
    size_t nen = (size_t)element->nen;
    size_t dof = (size_t)element->dof;
    element->nfix = 0;
    ierr = PetscMalloc1(nen*dof,&element->ifix);CHKERRQ(ierr);
    ierr = PetscMalloc1(nen*dof,&element->vfix);CHKERRQ(ierr);
    ierr = PetscMalloc1(nen*dof,&element->ufix);CHKERRQ(ierr);
    element->nflux = 0;
    ierr = PetscMalloc1(nen*dof,&element->iflux);CHKERRQ(ierr);
    ierr = PetscMalloc1(nen*dof,&element->vflux);CHKERRQ(ierr);
  }
  ierr = IGAPointInit(element->iterator,element);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGAGetElement(IGA iga,IGAElement *element)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(element,2);
  *element = iga->iterator;
  PetscFunctionReturn(0);
}

PetscErrorCode IGABeginElement(IGA iga,IGAElement *_element)
{
  IGAElement     element;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(_element,2);
  IGACheckSetUp(iga,1);
  element = *_element = iga->iterator;

  { /* */
    PetscInt dim = iga->dim, dof = iga->dof;
    PetscInt nsd = iga->geometry ? iga->geometry : iga->dim;
    PetscInt npd = iga->property ? iga->property : 0;
    if (element->dim != dim || element->dof != dof ||
        element->nsd != nsd || element->npd != npd)
      {ierr = IGAElementInit(element,iga);CHKERRQ(ierr);}
  }

  element->index = -1;
  element->atboundary  = PETSC_FALSE;
  element->boundary_id = -1;

  if (iga->rational && !iga->rationalW) SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_ARG_WRONGSTATE,"No geometry set");
  if (iga->geometry && !iga->geometryX) SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_ARG_WRONGSTATE,"No geometry set");
  if (iga->property && !iga->propertyA) SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_ARG_WRONGSTATE,"No property set");
  element->rational = iga->rational ? PETSC_TRUE : PETSC_FALSE;
  element->geometry = iga->geometry ? PETSC_TRUE : PETSC_FALSE;
  element->property = iga->property ? PETSC_TRUE : PETSC_FALSE;
  { /* */
    size_t nen = (size_t)element->nen;
    size_t nsd = iga->geometry ? (size_t)iga->geometry : (size_t)iga->dim;
    size_t npd = (size_t)iga->property;
    ierr = PetscMemzero(element->rationalW,sizeof(PetscReal)*nen);CHKERRQ(ierr);
    ierr = PetscMemzero(element->geometryX,sizeof(PetscReal)*nen*nsd);CHKERRQ(ierr);
    ierr = PetscMemzero(element->propertyA,sizeof(PetscReal)*nen*npd);CHKERRQ(ierr);
  }
  { /* */
    size_t q,i;
    size_t nqp = (size_t)element->nqp;
    size_t nen = (size_t)element->nen;
    size_t dim = (size_t)element->dim;
    size_t nsd = (size_t)element->nsd;
    /* */
    ierr = PetscMemzero(element->weight,sizeof(PetscReal)*nqp);CHKERRQ(ierr);
    ierr = PetscMemzero(element->detJac,sizeof(PetscReal)*nqp);CHKERRQ(ierr);
    /* */
    ierr = PetscMemzero(element->basis[0],sizeof(PetscReal)*nqp*nen);CHKERRQ(ierr);
    ierr = PetscMemzero(element->basis[1],sizeof(PetscReal)*nqp*nen*dim);CHKERRQ(ierr);
    ierr = PetscMemzero(element->basis[2],sizeof(PetscReal)*nqp*nen*dim*dim);CHKERRQ(ierr);
    ierr = PetscMemzero(element->basis[3],sizeof(PetscReal)*nqp*nen*dim*dim*dim);CHKERRQ(ierr);
    ierr = PetscMemzero(element->basis[4],sizeof(PetscReal)*nqp*nen*dim*dim*dim*dim);CHKERRQ(ierr);
    /* */
    ierr = PetscMemzero(element->shape[0],sizeof(PetscReal)*nqp*nen);CHKERRQ(ierr);
    ierr = PetscMemzero(element->shape[1],sizeof(PetscReal)*nqp*nen*nsd);CHKERRQ(ierr);
    ierr = PetscMemzero(element->shape[2],sizeof(PetscReal)*nqp*nen*nsd*nsd);CHKERRQ(ierr);
    ierr = PetscMemzero(element->shape[3],sizeof(PetscReal)*nqp*nen*nsd*nsd*nsd);CHKERRQ(ierr);
    ierr = PetscMemzero(element->shape[4],sizeof(PetscReal)*nqp*nen*nsd*nsd*nsd*nsd);CHKERRQ(ierr);
    /* */
    ierr = PetscMemzero(element->mapU[0],sizeof(PetscReal)*nqp*dim);CHKERRQ(ierr);
    ierr = PetscMemzero(element->mapU[1],sizeof(PetscReal)*nqp*dim*nsd);CHKERRQ(ierr);
    ierr = PetscMemzero(element->mapU[2],sizeof(PetscReal)*nqp*dim*nsd*nsd);CHKERRQ(ierr);
    ierr = PetscMemzero(element->mapU[3],sizeof(PetscReal)*nqp*dim*nsd*nsd*nsd);CHKERRQ(ierr);
    ierr = PetscMemzero(element->mapU[4],sizeof(PetscReal)*nqp*dim*nsd*nsd*nsd*nsd);CHKERRQ(ierr);
    /* */
    ierr = PetscMemzero(element->mapX[0],sizeof(PetscReal)*nqp*nsd);CHKERRQ(ierr);
    ierr = PetscMemzero(element->mapX[1],sizeof(PetscReal)*nqp*nsd*dim);CHKERRQ(ierr);
    ierr = PetscMemzero(element->mapX[2],sizeof(PetscReal)*nqp*nsd*dim*dim);CHKERRQ(ierr);
    ierr = PetscMemzero(element->mapX[3],sizeof(PetscReal)*nqp*nsd*dim*dim*dim);CHKERRQ(ierr);
    ierr = PetscMemzero(element->mapX[4],sizeof(PetscReal)*nqp*nsd*dim*dim*dim*dim);CHKERRQ(ierr);
    /* */
    ierr = PetscMemzero(element->normal,sizeof(PetscReal)*nqp*nsd);CHKERRQ(ierr);
    ierr = PetscMemzero(element->detX,sizeof(PetscReal)*nqp);CHKERRQ(ierr);
    ierr = PetscMemzero(element->detS,sizeof(PetscReal)*nqp);CHKERRQ(ierr);
    /* */
    if (!element->geometry)
      for (q=0; q<nqp; q++) {
        PetscReal *dX = &element->detX[q];
        PetscReal *X1 = &element->mapX[1][q*nsd*dim];
        PetscReal *U1 = &element->mapU[1][q*dim*nsd];
        dX[0] = 1.0;
        for (i=0; i<dim; i++) X1[i*(dim+1)] = 1.0;
        for (i=0; i<dim; i++) U1[i*(dim+1)] = 1.0;
      }
  }
  { /* */
    IGAPoint point = element->iterator;
    point->neq = element->neq;
    point->nen = element->nen;
    point->dof = element->dof;
    point->dim = element->dim;
    point->nsd = element->nsd;
    point->npd = element->npd;
    point->rational = element->rational ? element->rationalW : NULL;
    point->geometry = element->geometry ? element->geometryX : NULL;
    point->property = element->property ? element->propertyA : NULL;
  }
  PetscFunctionReturn(0);
}

PetscBool IGANextElement(PETSC_UNUSED IGA iga,IGAElement element)
{
  PetscInt i,dim  = element->dim;
  PetscInt *start = element->start;
  PetscInt *width = element->width;
  PetscInt *ID    = element->ID;
  PetscInt index,coord;

  PetscFunctionBegin;
  element->nval = 0;
  element->nvec = 0;
  element->nmat = 0;

  index = ++element->index;
  if (PetscUnlikely(index >= element->count)) goto stop;

  for (i=0; i<dim; i++) {
    coord = index % width[i];
    index = (index - coord) / width[i];
    ID[i] = coord + start[i];
  }

  {
    PetscErrorCode ierr;
    ierr = IGAElementBuildClosure(element);CHKERRCONTINUE(ierr);
    if (PetscUnlikely(ierr)) PetscFunctionReturn(PETSC_FALSE);
    ierr = IGAElementBuildFix(element);CHKERRCONTINUE(ierr);
    if (PetscUnlikely(ierr)) PetscFunctionReturn(PETSC_FALSE);
  }
  PetscFunctionReturn(PETSC_TRUE);

 stop:

  element->index = -1;
  PetscFunctionReturn(PETSC_FALSE);
}

PetscErrorCode IGAEndElement(IGA iga,IGAElement *element)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(element,2);
  PetscValidPointer(*element,2);
  if (PetscUnlikely((*element)->index != -1)) {
    (*element)->index = -1;
    *element = NULL;
    PetscFunctionReturn(PETSC_ERR_PLIB);
  }
  *element = NULL;
  PetscFunctionReturn(0);
}

PetscBool IGAElementNextForm(IGAElement element,PetscBool visit[][2])
{
  PetscInt dim = element->dim;
  PetscFunctionBegin;
  while (++element->boundary_id < 2*dim) {
    PetscInt i = element->boundary_id / 2;
    PetscInt s = element->boundary_id % 2;
    PetscInt e = s ? element->sizes[i]-1 : 0;
    if (element->ID[i] != e) continue;
    if (!visit[i][s]) continue;
    element->atboundary = PETSC_TRUE;
    PetscFunctionReturn(PETSC_TRUE);
  }
  if (element->boundary_id++ == 2*dim) {
    element->atboundary = PETSC_FALSE;
    PetscFunctionReturn(PETSC_TRUE);
  }
  element->atboundary  = PETSC_FALSE;
  element->boundary_id = -1;
  PetscFunctionReturn(PETSC_FALSE);
}

PetscErrorCode IGAElementGetPoint(IGAElement element,IGAPoint *point)
{
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  PetscValidPointer(point,2);
  *point = element->iterator;
  PetscFunctionReturn(0);
}

PetscErrorCode IGAElementBeginPoint(IGAElement element,IGAPoint *_point)
{
  IGAPoint       point;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(element,1);
  PetscValidPointer(_point,2);
  if (PetscUnlikely(element->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during element loop");
  point = *_point = element->iterator;

  point->index = -1;
  point->atboundary  = element->atboundary;
  point->boundary_id = element->boundary_id;
  ierr = IGAElementBuildTabulation(element);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscBool IGAElementNextPoint(IGAElement element,IGAPoint point)
{
  PetscInt nen  = point->nen;
  PetscInt dim  = point->dim;
  PetscInt nsd  = point->nsd;
  PetscInt dim1 = dim;
  PetscInt dim2 = dim*dim1;
  PetscInt dim3 = dim*dim2;
  PetscInt dim4 = dim*dim3;
  PetscInt nsd1 = nsd;
  PetscInt nsd2 = nsd*nsd1;
  PetscInt nsd3 = nsd*nsd2;
  PetscInt nsd4 = nsd*nsd3;
  PetscInt index;

  PetscFunctionBegin;
  point->nvec = 0;
  point->nmat = 0;

  index = ++point->index;
  if (PetscUnlikely(index == 0))            goto start;
  if (PetscUnlikely(index >= point->count)) goto stop;

  point->weight   += 1;
  point->detJac   += 1;

  point->point    += dim;
  point->normal   += nsd;

  point->basis[0] += nen;
  point->basis[1] += nen*dim1;
  point->basis[2] += nen*dim2;
  point->basis[3] += nen*dim3;
  point->basis[4] += nen*dim4;

  point->shape[0] += nen;
  point->shape[1] += nen*nsd1;
  point->shape[2] += nen*nsd2;
  point->shape[3] += nen*nsd3;
  point->shape[4] += nen*nsd4;

  point->mapU[0]  += dim;
  point->mapU[1]  += dim*nsd1;
  point->mapU[2]  += dim*nsd2;
  point->mapU[3]  += dim*nsd3;
  point->mapU[4]  += dim*nsd4;

  point->mapX[0]  += nsd;
  point->mapX[1]  += nsd*dim1;
  point->mapX[2]  += nsd*dim2;
  point->mapX[3]  += nsd*dim3;
  point->mapX[4]  += nsd*dim4;

  point->detX     += 1;
  point->detS     += 1;

  PetscFunctionReturn(PETSC_TRUE);

 start:

  point->weight = element->weight;
  point->detJac = element->detJac;

  point->point  = element->mapU[0];
  point->normal = element->normal;

  point->basis[0] = element->basis[0];
  point->basis[1] = element->basis[1];
  point->basis[2] = element->basis[2];
  point->basis[3] = element->basis[3];
  point->basis[4] = element->basis[4];

  if (element->geometry && dim == nsd) {
    point->shape[0] = element->shape[0];
    point->shape[1] = element->shape[1];
    point->shape[2] = element->shape[2];
    point->shape[3] = element->shape[3];
    point->shape[4] = element->shape[4];
  } else {
    point->shape[0] = element->basis[0];
    point->shape[1] = element->basis[1];
    point->shape[2] = element->basis[2];
    point->shape[3] = element->basis[3];
    point->shape[4] = element->basis[4];
  }

  point->mapU[0] = element->mapU[0];
  point->mapU[1] = element->mapU[1];
  point->mapU[2] = element->mapU[2];
  point->mapU[3] = element->mapU[3];
  point->mapU[4] = element->mapU[4];

  if (element->geometry) {
    point->mapX[0] = element->mapX[0];
    point->mapX[1] = element->mapX[1];
    point->mapX[2] = element->mapX[2];
    point->mapX[3] = element->mapX[3];
    point->mapX[4] = element->mapX[4];
  } else {
    point->mapX[0] = element->mapU[0];
    point->mapX[1] = element->mapU[1];
    point->mapX[2] = element->mapU[2];
    point->mapX[3] = element->mapU[3];
    point->mapX[4] = element->mapU[4];
  }

  point->detX = element->detX;
  point->detS = element->detS;

  PetscFunctionReturn(PETSC_TRUE);

 stop:

  point->index = -1;
  PetscFunctionReturn(PETSC_FALSE);
}

PetscErrorCode IGAElementEndPoint(IGAElement element,IGAPoint *point)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  PetscValidPointer(point,2);
  PetscValidPointer(*point,2);
  if (PetscUnlikely((*point)->index != -1)) {
    (*point)->index = -1;
    PetscFunctionReturn(PETSC_ERR_PLIB);
  }
  *point = NULL;
  /* XXX */
  if (PetscUnlikely(element->atboundary)) {
    size_t    nqp = (size_t)element->nqp;
    size_t    nsd = (size_t)element->nsd;
    PetscReal *dS = element->detS;
    PetscReal *n  = element->normal;
    ierr = PetscMemzero(dS,nqp*sizeof(PetscReal));CHKERRQ(ierr);
    ierr = PetscMemzero(n,nqp*nsd*sizeof(PetscReal));CHKERRQ(ierr);
    if (PetscUnlikely(element->collocation)) {
      element->atboundary  = PETSC_FALSE;
      element->boundary_id = 2*element->dim;
    }
  }
  /* XXX */
 PetscFunctionReturn(0);
}

PetscErrorCode IGAElementGetParent(IGAElement element,IGA *parent)
{
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  PetscValidIntPointer(parent,2);
  *parent = element->parent;
  PetscFunctionReturn(0);
}

PetscErrorCode IGAElementGetIndex(IGAElement element,PetscInt *index)
{
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  PetscValidIntPointer(index,2);
  *index = element->index;
  PetscFunctionReturn(0);
}

PetscErrorCode IGAElementGetCount(IGAElement element,PetscInt *count)
{
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  PetscValidIntPointer(count,2);
  *count = element->count;
  PetscFunctionReturn(0);
}

PetscErrorCode IGAElementGetSizes(IGAElement element,PetscInt *neq,PetscInt *nen,PetscInt *dof)
{
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  if (neq) PetscValidIntPointer(neq,2);
  if (nen) PetscValidIntPointer(nen,3);
  if (dof) PetscValidIntPointer(dof,4);
  if (neq) *neq = element->neq;
  if (nen) *nen = element->nen;
  if (dof) *dof = element->dof;
  PetscFunctionReturn(0);
}

PetscErrorCode IGAElementGetClosure(IGAElement element,PetscInt *nen,const PetscInt *mapping[])
{
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  if (nen)     PetscValidIntPointer(nen,2);
  if (mapping) PetscValidPointer(mapping,3);
  if (nen)     *nen     = element->nen;
  if (mapping) *mapping = element->mapping;
  PetscFunctionReturn(0);
}

PetscErrorCode IGAElementGetIndices(IGAElement element,
                                    PetscInt *neq,const PetscInt *rowmap[],
                                    PetscInt *nen,const PetscInt *colmap[])
{
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  if (neq)    PetscValidIntPointer(neq,2);
  if (rowmap) PetscValidPointer(rowmap,3);
  if (nen)    PetscValidIntPointer(nen,4);
  if (colmap) PetscValidPointer(colmap,5);
  if (neq)    *neq    = element->neq;
  if (colmap) *colmap = element->rowmap;
  if (nen)    *nen    = element->nen;
  if (colmap) *colmap = element->colmap;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

PetscErrorCode IGAElementBuildClosure(IGAElement element)
{
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  if (PetscUnlikely(element->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during element loop");
  { /* */
    IGA      iga = element->parent;
    IGABasis *BD = element->parent->basis;
    PetscInt *ID = element->ID;
    PetscInt ia, inen = BD[0]->nen, ioffset = BD[0]->offset[ID[0]];
    PetscInt ja, jnen = BD[1]->nen, joffset = BD[1]->offset[ID[1]];
    PetscInt ka, knen = BD[2]->nen, koffset = BD[2]->offset[ID[2]];
    PetscInt *start = iga->node_gstart, *width = iga->node_gwidth;
    PetscInt istart = start[0]/*istride = 1*/;
    PetscInt jstart = start[1], jstride = width[0];
    PetscInt kstart = start[2], kstride = width[0]*width[1];
    PetscInt a=0, *mapping = element->mapping;
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
    if (PetscUnlikely(element->collocation)) {
      PetscInt iA = ID[0] - istart;
      PetscInt jA = ID[1] - jstart;
      PetscInt kA = ID[2] - kstart;
      element->rowmap[0] = iA + jA*jstride + kA*kstride;
    }
  }
  { /* */
    PetscInt a,nen = element->nen;
    PetscInt *map = element->mapping;
    if (element->rational) {
      PetscReal *arrayW = element->parent->rationalW;
      PetscReal *W = element->rationalW;
      for (a=0; a<nen; a++)
        W[a] = arrayW[map[a]];
    }
    if (element->geometry) {
      PetscReal *arrayX = element->parent->geometryX;
      PetscReal *X = element->geometryX;
      PetscInt  i,nsd = element->nsd;
      for (a=0; a<nen; a++)
        for (i=0; i<nsd; i++)
          X[i + a*nsd] = arrayX[map[a]*nsd + i];
    }
    if (element->property) {
      PetscScalar *arrayA = element->parent->propertyA;
      PetscScalar *A = element->propertyA;
      PetscInt    i,npd = element->npd;
      for (a=0; a<nen; a++)
        for (i=0; i<npd; i++)
          A[i + a*npd] = arrayA[map[a]*npd + i];
    }
  }
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */

#include "petigaftn.h"

EXTERN_C_BEGIN
extern void IGA_GetNormal(PetscInt dim,PetscInt axis,PetscInt side,const PetscReal F[],PetscReal *dS,PetscReal n[]);
EXTERN_C_END

static inline
PetscInt IGA_Quadrature_SIZE(const IGABasis BD[],const PetscInt ID[],PetscInt NQ[3])
{
  PetscInt i;
  for (i=0; i<3; i++) {
    PetscInt q = BD[i]->nqp - 1;
    PetscReal *w = BD[i]->weight + ID[i]*BD[i]->nqp;
    NQ[i] = 1;
    while (q >= 0 && w[q] <= 0) q--;
    NQ[i] += q;
  }
  return NQ[0]*NQ[1]*NQ[2];
}

#define IGA_Quadrature_ARGS(BD,ID,NQ,i) \
  NQ[i],                                \
  BD[i]->point  + ID[i]*BD[i]->nqp,     \
  BD[i]->weight + ID[i]*BD[i]->nqp,     \
  BD[i]->detJac + ID[i]

#define IGA_BasisFuns_ARGS(BD,ID,NQ,i)  \
  NQ[i], BD[i]->nen,                    \
  BD[i]->value + ID[i]*BD[i]->nqp*BD[i]->nen*5

#define IGA_Quadrature_BNDR(BD,ID,NQ,i,s) \
  1,&BD[i]->bnd_point[s],&BD[i]->bnd_weight,&BD[i]->bnd_detJac

#define IGA_BasisFuns_BNDR(BD,ID,NQ,i,s) \
  1,BD[i]->nen,BD[i]->bnd_value[s]

PetscErrorCode IGAElementBuildTabulation(IGAElement element)
{
  PetscInt axis = PETSC_MAX_INT;
  PetscInt side = PETSC_MAX_INT;
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  if (PetscUnlikely(element->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during element loop");
  {
    PetscInt ord = element->parent->order;
    PetscInt dim = element->dim;
    PetscInt nsd = element->nsd;
    PetscInt nen = element->nen;
    IGABasis *BD = element->parent->basis;
    PetscInt *ID = element->ID;
    PetscInt *NQ = element->sqp;
    PetscInt nqp,q;

    nqp = IGA_Quadrature_SIZE(BD,ID,NQ);
    if (PetscUnlikely(element->atboundary)) {
      axis = element->boundary_id / 2;
      side = element->boundary_id % 2;
      nqp /= NQ[axis]; NQ[axis] = 1;
    }
    element->iterator->count = nqp; /* XXX */

    {
      PetscReal *u = element->mapU[0];
      PetscReal *w = element->weight;
      PetscReal *J = element->detJac;
      if (PetscLikely(!element->atboundary))
        switch (dim) {
        case 3: IGA_Quadrature_3D(IGA_Quadrature_ARGS(BD,ID,NQ,0),
                                  IGA_Quadrature_ARGS(BD,ID,NQ,1),
                                  IGA_Quadrature_ARGS(BD,ID,NQ,2),
                                  u,w,J); break;
        case 2: IGA_Quadrature_2D(IGA_Quadrature_ARGS(BD,ID,NQ,0),
                                  IGA_Quadrature_ARGS(BD,ID,NQ,1),
                                  u,w,J); break;
        case 1: IGA_Quadrature_1D(IGA_Quadrature_ARGS(BD,ID,NQ,0),
                                  u,w,J); break;
        }
      else
        switch (dim) {
        case 3:
          switch (axis) {
          case 0: IGA_Quadrature_3D(IGA_Quadrature_BNDR(BD,ID,NQ,0,side),
                                    IGA_Quadrature_ARGS(BD,ID,NQ,1),
                                    IGA_Quadrature_ARGS(BD,ID,NQ,2),
                                    u,w,J); break;
          case 1: IGA_Quadrature_3D(IGA_Quadrature_ARGS(BD,ID,NQ,0),
                                    IGA_Quadrature_BNDR(BD,ID,NQ,1,side),
                                    IGA_Quadrature_ARGS(BD,ID,NQ,2),
                                    u,w,J); break;
          case 2: IGA_Quadrature_3D(IGA_Quadrature_ARGS(BD,ID,NQ,0),
                                    IGA_Quadrature_ARGS(BD,ID,NQ,1),
                                    IGA_Quadrature_BNDR(BD,ID,NQ,2,side),
                                    u,w,J); break;
          } break;
        case 2:
          switch (axis) {
          case 0: IGA_Quadrature_2D(IGA_Quadrature_BNDR(BD,ID,NQ,0,side),
                                    IGA_Quadrature_ARGS(BD,ID,NQ,1),
                                    u,w,J); break;
          case 1: IGA_Quadrature_2D(IGA_Quadrature_ARGS(BD,ID,NQ,0),
                                    IGA_Quadrature_BNDR(BD,ID,NQ,1,side),
                                    u,w,J); break;
          } break;
        case 1:
          switch (axis) {
          case 0: IGA_Quadrature_1D(IGA_Quadrature_BNDR(BD,ID,NQ,0,side),
                                    u,w,J); break;
          } break;
        }
    }

    {
      PetscReal **M = element->basis;
      if (PetscLikely(!element->atboundary))
        switch (dim) {
        case 3: IGA_BasisFuns_3D(ord,
                                 IGA_BasisFuns_ARGS(BD,ID,NQ,0),
                                 IGA_BasisFuns_ARGS(BD,ID,NQ,1),
                                 IGA_BasisFuns_ARGS(BD,ID,NQ,2),
                                 M[0],M[1],M[2],M[3],M[4]); break;
        case 2: IGA_BasisFuns_2D(ord,
                                 IGA_BasisFuns_ARGS(BD,ID,NQ,0),
                                 IGA_BasisFuns_ARGS(BD,ID,NQ,1),
                                 M[0],M[1],M[2],M[3],M[4]); break;
        case 1: IGA_BasisFuns_1D(ord,
                                 IGA_BasisFuns_ARGS(BD,ID,NQ,0),
                                 M[0],M[1],M[2],M[3],M[4]); break;
        }
      else
        switch (dim) {
        case 3:
          switch (axis) {
          case 0: IGA_BasisFuns_3D(ord,
                                   IGA_BasisFuns_BNDR(BD,ID,NQ,0,side),
                                   IGA_BasisFuns_ARGS(BD,ID,NQ,1),
                                   IGA_BasisFuns_ARGS(BD,ID,NQ,2),
                                   M[0],M[1],M[2],M[3],M[4]); break;
          case 1: IGA_BasisFuns_3D(ord,
                                   IGA_BasisFuns_ARGS(BD,ID,NQ,0),
                                   IGA_BasisFuns_BNDR(BD,ID,NQ,1,side),
                                   IGA_BasisFuns_ARGS(BD,ID,NQ,2),
                                   M[0],M[1],M[2],M[3],M[4]); break;
          case 2: IGA_BasisFuns_3D(ord,
                                   IGA_BasisFuns_ARGS(BD,ID,NQ,0),
                                   IGA_BasisFuns_ARGS(BD,ID,NQ,1),
                                   IGA_BasisFuns_BNDR(BD,ID,NQ,2,side),
                                   M[0],M[1],M[2],M[3],M[4]); break;
          } break;
        case 2:
          switch (axis) {
          case 0: IGA_BasisFuns_2D(ord,
                                   IGA_BasisFuns_BNDR(BD,ID,NQ,0,side),
                                   IGA_BasisFuns_ARGS(BD,ID,NQ,1),
                                   M[0],M[1],M[2],M[3],M[4]); break;
          case 1: IGA_BasisFuns_2D(ord,
                                   IGA_BasisFuns_ARGS(BD,ID,NQ,0),
                                   IGA_BasisFuns_BNDR(BD,ID,NQ,1,side),
                                   M[0],M[1],M[2],M[3],M[4]); break;
          } break;
        case 1:
          switch (axis) {
          case 0: IGA_BasisFuns_1D(ord,
                                   IGA_BasisFuns_BNDR(BD,ID,NQ,0,side),
                                   M[0],M[1],M[2],M[3],M[4]); break;
          } break;
        }
    }

    if (element->rational) {
      PetscReal *W  = element->rationalW;
      PetscReal **M = element->basis;
      switch (dim) {
      case 3: IGA_Rationalize_3D(ord,nqp,nen,W,
                                 M[0],M[1],M[2],M[3],M[4]); break;
      case 2: IGA_Rationalize_2D(ord,nqp,nen,W,
                                 M[0],M[1],M[2],M[3],M[4]); break;
      case 1: IGA_Rationalize_1D(ord,nqp,nen,W,
                                 M[0],M[1],M[2],M[3],M[4]); break;
      }
    }

    if (element->geometry) {
      PetscReal *X  = element->geometryX;
      PetscReal **M = element->basis;
      PetscReal *X0 = element->mapX[0];
      PetscReal *X1 = element->mapX[1];
      PetscReal *X2 = element->mapX[2];
      PetscReal *X3 = element->mapX[3];
      PetscReal *X4 = element->mapX[4];
      if (PetscLikely(dim == nsd))
        switch (dim) {
        case 3: IGA_GeometryMap_3D(ord,nqp,nen,X,
                                   M[0],M[1],M[2],M[3],M[4],
                                   X0,X1,X2,X3,X4); break;
        case 2: IGA_GeometryMap_2D(ord,nqp,nen,X,
                                   M[0],M[1],M[2],M[3],M[4],
                                   X0,X1,X2,X3,X4); break;
        case 1: IGA_GeometryMap_1D(ord,nqp,nen,X,
                                   M[0],M[1],M[2],M[3],M[4],
                                   X0,X1,X2,X3,X4); break;
        }
      else
        IGA_GeometryMap(ord,dim,nsd,nqp,nen,X,
                        M[0],M[1],M[2],M[3],M[4],
                        X0,X1,X2,X3,X4);
    }

    if (element->geometry && PetscLikely(dim == nsd)) {
      PetscReal **M = element->basis;
      PetscReal **N = element->shape;
      PetscReal *dX = element->detX;
      PetscReal *X1 = element->mapX[1];
      PetscReal *X2 = element->mapX[2];
      PetscReal *X3 = element->mapX[3];
      PetscReal *X4 = element->mapX[4];
      PetscReal *U1 = element->mapU[1];
      PetscReal *U2 = element->mapU[2];
      PetscReal *U3 = element->mapU[3];
      PetscReal *U4 = element->mapU[4];
      switch (dim) {
      case 3: IGA_InverseMap_3D(ord,nqp,
                                X1,X2,X3,X4,dX,
                                U1,U2,U3,U4); break;
      case 2: IGA_InverseMap_2D(ord,nqp,
                                X1,X2,X3,X4,dX,
                                U1,U2,U3,U4); break;
      case 1: IGA_InverseMap_1D(ord,nqp,
                                X1,X2,X3,X4,dX,
                                U1,U2,U3,U4); break;
      }
#if defined(PETSC_USE_DEBUG)
      for (q=0; q<nqp; q++)
        if (dX[q] <= 0)
          SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Non-positive det(Jacobian)=%g",(double)dX[q]);
#endif
      /* */
      (void)PetscMemcpy(N[0],M[0],(size_t)(nqp*nen)*sizeof(PetscReal));
      switch (dim) {
      case 3: IGA_ShapeFuns_3D(ord,nqp,nen,
                               U1,U2,U3,U4,
                               M[1],M[2],M[3],M[4],
                               N[1],N[2],N[3],N[4]); break;
      case 2: IGA_ShapeFuns_2D(ord,nqp,nen,
                               U1,U2,U3,U4,
                               M[1],M[2],M[3],M[4],
                               N[1],N[2],N[3],N[4]); break;
      case 1: IGA_ShapeFuns_1D(ord,nqp,nen,
                               U1,U2,U3,U4,
                               M[1],M[2],M[3],M[4],
                               N[1],N[2],N[3],N[4]); break;
      }
    }

    if (PetscUnlikely(element->atboundary)) {
      PetscReal *X1 = element->mapX[1];
      PetscReal *dS = element->detS;
      PetscReal *n  = element->normal;
      if (element->geometry && PetscLikely(dim == nsd)) {
        for (q=0; q<nqp; q++) IGA_GetNormal(dim,axis,side,&X1[q*nsd*dim],&dS[q],&n[q*nsd]);
      } else {
        (void)PetscMemzero(n,(size_t)(nqp*nsd)*sizeof(PetscReal));
        for (q=0; q<nqp; q++) {dS[q] = 1.0; n[q*nsd+axis] = side ? 1.0 : -1.0;}
      }
    }

    if (element->geometry && PetscLikely(dim == nsd) && PetscLikely(!element->collocation)) {
      if (PetscLikely(!element->atboundary))
        for (q=0; q<nqp; q++) element->detJac[q] *= element->detX[q];
      else
        for (q=0; q<nqp; q++) element->detJac[q] *= element->detS[q];
    }

  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

PetscErrorCode IGAElementGetWorkVec(IGAElement element,PetscScalar *V[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  PetscValidPointer(V,2);
  if (PetscUnlikely(element->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during element loop");
  if (PetscUnlikely((size_t)element->nvec >= sizeof(element->wvec)/sizeof(PetscScalar*)))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many work vectors requested");
  {
    size_t m = (size_t)element->neq * (size_t)element->dof;
    *V = element->wvec[element->nvec++];
    ierr = PetscMemzero(*V,m*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IGAElementGetWorkMat(IGAElement element,PetscScalar *M[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  PetscValidPointer(M,2);
  if (PetscUnlikely(element->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during element loop");
  if (PetscUnlikely((size_t)element->nmat >= sizeof(element->wmat)/sizeof(PetscScalar*)))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many work matrices requested");
  {
    size_t m = (size_t)element->neq * (size_t)element->dof;
    size_t n = (size_t)element->nen * (size_t)element->dof;
    *M = element->wmat[element->nmat++];
    ierr = PetscMemzero(*M,m*n*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IGAElementGetValues(IGAElement element,const PetscScalar arrayU[],PetscScalar *_U[])
{
  PetscScalar    *U;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  if (arrayU) PetscValidScalarPointer(arrayU,2);
  PetscValidPointer(_U,3);
  if (PetscUnlikely(element->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during element loop");
  if (PetscUnlikely((size_t)element->nval >= sizeof(element->wval)/sizeof(PetscScalar*)))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many work values requested");
  U = *_U = element->wval[element->nval++];
  if (PetscLikely(arrayU)) {
    PetscInt a, nen = element->nen;
    PetscInt i, dof = element->dof;
    PetscInt pos = 0, *map = element->mapping;
    for (a=0; a<nen; a++) {
      const PetscScalar *u = arrayU + map[a]*dof;
      for (i=0; i<dof; i++) U[pos++] = u[i]; /* XXX Use PetscMemcpy() ?? */
    }
  } else {
    size_t n = (size_t)(element->nen * element->dof);
    ierr = PetscMemzero(U,n*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

EXTERN_C_BEGIN
extern void IGA_BoundaryArea_2D(const PetscInt[],PetscInt,PetscInt,
                                PetscInt,const PetscReal[],
                                PetscInt,const PetscReal[],
                                PetscInt,const PetscReal[],PetscInt,const PetscReal[],
                                PetscReal*);
extern void IGA_BoundaryArea_3D(const PetscInt[],PetscInt,PetscInt,
                                PetscInt,const PetscReal[],
                                PetscInt,const PetscReal[],
                                PetscInt,const PetscReal[],PetscInt,const PetscReal[],
                                PetscInt,const PetscReal[],PetscInt,const PetscReal[],
                                PetscReal*);
EXTERN_C_END

static PetscReal BoundaryArea(IGAElement element,PetscInt dir,PetscInt side)
{
  PetscReal A = 1;
  PetscInt *ID = element->ID;
  IGABasis *BD = element->parent->basis;
  PetscInt i,dim = element->dim;
  if (dim == 1) return A;
  for (i=0; i<dim; i++)
    if (i != dir) {
      PetscReal L = BD[i]->detJac[ID[i]];
      PetscInt  n = BD[i]->nen;
      A *= L/(PetscReal)n;
    }
  if (!element->geometry) {
    A *= (dim==2) ? 2 : 4; /* sum(W) = 2 */
  } else {
    PetscInt qshape[3] = {1,1,1};
    PetscInt nshape[3] = {1,1,1};
    PetscInt k,nqp[3],nen[3];
    PetscReal *W[3],*N[3],dS = 1;
    (void)IGA_Quadrature_SIZE(BD,ID,qshape);
    for (i=0; i<dim; i++) nshape[i] = BD[i]->nen;
    for (k=0,i=0; i<dim; i++) {
      if (i == dir) continue;
      nqp[k] = qshape[i];
      nen[k] = nshape[i];
      W[k] = BD[i]->weight + ID[i]*BD[i]->nqp;
      N[k] = BD[i]->value  + ID[i]*BD[i]->nqp*nen[k]*5;
      k++;
    }
    switch (dim) {
    case 2: IGA_BoundaryArea_2D(nshape,dir,side,
                                element->geometry,element->geometryX,
                                element->rational,element->rationalW,
                                nqp[0],W[0],nen[0],N[0],
                                &dS); break;
    case 3: IGA_BoundaryArea_3D(nshape,dir,side,
                                element->geometry,element->geometryX,
                                element->rational,element->rationalW,
                                nqp[0],W[0],nen[0],N[0],
                                nqp[1],W[1],nen[1],N[1],
                                &dS);break;
    }
    A *= dS;
  }
  return A;
}

static void AddFixa(IGAElement element,IGAFormBC bc,PetscInt a)
{
  if (bc->count) {
    IGA iga = element->parent;
    PetscInt dof = element->dof;
    PetscInt count = element->nfix;
    PetscInt *index = element->ifix;
    PetscScalar *value = element->vfix;
    PetscInt j,k,n = bc->count;
    for (k=0; k<n; k++) {
      PetscInt c = bc->field[k];
      PetscInt idx = a*dof + c;
      PetscScalar val = bc->value[k];
      if (PetscUnlikely(c >= dof)) continue;
      if (iga->fixtable) val = iga->fixtableU[c + element->mapping[a]*dof];
      for (j=0; j<count; j++)
        if (index[j] == idx) break;
      if (j == count) count++;
      index[j] = idx;
      value[j] = val;
    }
    element->nfix = count;
  }
}

static void AddFlux(IGAElement element,IGAFormBC bc,PetscInt a,PetscReal A)
{
  if (bc->count) {
    PetscInt dof = element->dof;
    PetscInt count = element->nflux;
    PetscInt *index = element->iflux;
    PetscScalar *value = element->vflux;
    PetscInt j,k,n = bc->count;
    for (k=0; k<n; k++) {
      PetscInt c = bc->field[k];
      PetscInt idx = a*dof + c;
      PetscScalar val = bc->value[k];
      if (PetscUnlikely(c >= dof)) continue;
      for (j=0; j<count; j++)
        if (index[j] == idx) break;
      if (j == count) value[count++] = 0.0;
      index[j]  = idx;
      value[j] += val*A;
    }
    element->nflux = count;
  }
}

static void BuildFix(IGAElement element,PetscInt dir,PetscInt side)
{
  IGAForm   form = element->parent->form;
  IGAFormBC bcv  = form->value[dir][side];
  IGAFormBC bcl  = form->load [dir][side];
  if (bcv->count || bcl->count) {
    PetscReal Area = bcl->count ? BoundaryArea(element,dir,side) : 1;
    IGABasis *BD = element->parent->basis;
    PetscInt S[3]={0,0,0},E[3]={1,1,1};
    PetscInt ia,ja,ka,jstride,kstride,a;
    PetscInt i,dim = element->dim;
    for (i=0; i<dim; i++) E[i] = BD[i]->nen;
    jstride = E[0]; kstride = E[0]*E[1];
    if (side) S[dir] = E[dir]-1;
    else      E[dir] = S[dir]+1;
    for (ka=S[2]; ka<E[2]; ka++)
      for (ja=S[1]; ja<E[1]; ja++)
        for (ia=S[0]; ia<E[0]; ia++)
          {
            a = ia + ja*jstride + ka*kstride;
            AddFixa(element,bcv,a);
            AddFlux(element,bcl,a,Area);
          }
  }
}

static inline
IGAFormBC AtBoundaryV(IGAElement element,PetscInt dir,PetscInt side)
{
  IGAFormBC bc = element->parent->form->value[dir][side];
  PetscInt  e  = side ? element->sizes[dir]-1 : 0;
  return (element->ID[dir] == e) ? bc : NULL;
}
static inline
IGAFormBC AtBoundaryL(IGAElement element,PetscInt dir,PetscInt side)
{
  IGAFormBC bc = element->parent->form->load[dir][side];
  PetscInt  e  = side ? element->sizes[dir]-1 : 0;
  return (element->ID[dir] == e) ? bc : NULL;
}

static inline
PetscReal DOT(PetscInt dim,const PetscReal a[],const PetscReal b[])
{
  PetscInt i; PetscReal s = 0.0;
  for (i=0; i<dim; i++) s += a[i]*b[i];
  return s;
}

PetscErrorCode IGAElementBuildFix(IGAElement element)
{
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  if (PetscUnlikely(element->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during element loop");
  if (PetscUnlikely(element->collocation)) goto collocation;
  element->nfix  = 0;
  element->nflux = 0;
  {
    IGAAxis  *AX = element->parent->axis;
    PetscInt *ID = element->ID;
    PetscInt i,dim = element->dim;
    for (i=0; i<dim; i++) {
      PetscBool w = AX[i]->periodic;
      PetscInt  e = element->sizes[i]-1; /* last element */
      if (ID[i] == 0 && !w) BuildFix(element,i,0);
      if (ID[i] == e && !w) BuildFix(element,i,1);
    }
  }
  PetscFunctionReturn(0);
 collocation:
  element->nfix  = 0;
  element->nflux = 0;
  {
    PetscInt L[3] = {PETSC_MIN_INT,PETSC_MIN_INT,PETSC_MIN_INT};
    PetscInt R[3] = {PETSC_MAX_INT,PETSC_MAX_INT,PETSC_MAX_INT};
    {
      IGAAxis  *AX = element->parent->axis;
      PetscInt i,dim = element->dim;
      for (i=0; i<dim; i++) {
        PetscBool w = AX[i]->periodic;
        PetscInt  n = element->sizes[i]-1; /* last node */
        L[i] = 0; if (!w) R[i] = n;
      }
    }
    {
      IGAFormBC (*bc)[2] = element->parent->form->value;
      IGABasis *BD = element->parent->basis;
      PetscInt *ID = element->ID;
      PetscInt ia, inen = BD[0]->nen, ioffset = BD[0]->offset[ID[0]];
      PetscInt ja, jnen = BD[1]->nen, joffset = BD[1]->offset[ID[1]];
      PetscInt ka, knen = BD[2]->nen, koffset = BD[2]->offset[ID[2]];
      PetscInt a = 0;
      for (ka=0; ka<knen; ka++)
        for (ja=0; ja<jnen; ja++)
          for (ia=0; ia<inen; ia++)
            {
              PetscInt iA = ioffset + ia;
              PetscInt jA = joffset + ja;
              PetscInt kA = koffset + ka;
              /**/ if (iA == L[0]) AddFixa(element,bc[0][0],a);
              else if (iA == R[0]) AddFixa(element,bc[0][1],a);
              /**/ if (jA == L[1]) AddFixa(element,bc[1][0],a);
              else if (jA == R[1]) AddFixa(element,bc[1][1],a);
              /**/ if (kA == L[2]) AddFixa(element,bc[2][0],a);
              else if (kA == R[2]) AddFixa(element,bc[2][1],a);
              a++;
            }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IGAElementDelValues(IGAElement element,PetscScalar V[])
{
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  PetscValidScalarPointer(V,2);
  {
    PetscInt f,n,k;
    n = element->nfix;
    for (f=0; f<n; f++) {
      k = element->ifix[f];
      V[k] = (PetscScalar)0.0;
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IGAElementFixValues(IGAElement element,PetscScalar U[])
{
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  PetscValidScalarPointer(U,2);
  {
    PetscInt f,n,k;
    n = element->nfix;
    for (f=0; f<n; f++) {
      k = element->ifix[f];
      element->ufix[f] = U[k];
      U[k] = element->vfix[f];
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IGAElementFixSystem(IGAElement element,PetscScalar K[],PetscScalar F[])
{
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  PetscValidScalarPointer(K,2);
  PetscValidScalarPointer(F,3);
  if (PetscUnlikely(element->collocation)) goto collocation;
  {
    PetscInt M = element->neq * element->dof;
    PetscInt N = element->nen * element->dof;
    PetscInt f,n;
    n = element->nflux;
    for (f=0; f<n; f++) {
      PetscInt    k = element->iflux[f];
      PetscScalar v = element->vflux[f];
      F[k] += v;
    }
    n = element->nfix;
    for (f=0; f<n; f++) {
      PetscInt    k = element->ifix[f];
      PetscScalar v = element->vfix[f];
      PetscInt i,j;
      for (i=0; i<M; i++) F[i] -= K[i*N+k] * v;
      for (i=0; i<M; i++) K[i*N+k] = 0.0;
      for (j=0; j<N; j++) K[k*N+j] = 0.0;
      K[k*N+k] = 1.0;
      F[k]     = v;
    }
  }
  PetscFunctionReturn(0);
 collocation:
  {
    PetscInt dim = element->dim;
    PetscInt dof = element->dof;
    PetscInt nen = element->nen;
    PetscInt N = nen * dof;
    PetscInt dir,side;
    for (dir=0; dir<dim; dir++) {
      for (side=0; side<2; side++) {
        IGAFormBC bcl = AtBoundaryL(element,dir,side);
        IGAFormBC bcv = AtBoundaryV(element,dir,side);
        if (bcl && bcl->count) {
          PetscInt  f, n = bcl->count;
          PetscReal *dshape, normal[3] = {0,0,0};
          if (!element->geometry) {
            normal[dir] = side ? 1.0 : -1.0;
            dshape = element->basis[1];
          } else {
            PetscReal dS, *X1 = element->mapX[1];
            IGA_GetNormal(dim,dir,side,X1,&dS,normal);
            dshape = element->shape[1];
          }
          for (f=0; f<n; f++) {
            PetscInt    c = bcl->field[f];
            PetscScalar v = bcl->value[f];
            PetscInt    a,j;
            for (j=0; j<N; j++) K[c*N+j] = 0.0;
            for (a=0; a<nen; a++)
              K[c*N+a*dof+c] = DOT(dim,&dshape[a*dim],normal);
            F[c] = v;
          }
        }
        if (bcv && bcv->count) {
          PetscInt  f, n = bcv->count;
          PetscReal *shape = element->basis[0];
          for (f=0; f<n; f++) {
            PetscInt    c = bcv->field[f];
            PetscScalar v = bcv->value[f];
            PetscInt    a,j;
            for (j=0; j<N; j++) K[c*N+j] = 0.0;
            for (a=0; a<nen; a++)
              K[c*N+a*dof+c] = shape[a];
            F[c] = v;
          }
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IGAElementFixFunction(IGAElement element,PetscScalar F[])
{
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  PetscValidScalarPointer(F,2);
  if (PetscUnlikely(element->collocation)) goto collocation;
  {
    PetscInt f,n;
    n = element->nflux;
    for (f=0; f<n; f++) {
      PetscInt    k = element->iflux[f];
      PetscScalar v = element->vflux[f];
      F[k] -= v;
    }
    n = element->nfix;
    for (f=0; f<n; f++) {
      PetscInt    k = element->ifix[f];
      PetscScalar v = element->vfix[f];
      PetscScalar u = element->ufix[f];
      F[k] = u - v;
    }
  }
  PetscFunctionReturn(0);
 collocation:
  {
    PetscInt f,n;
    n = element->nfix;
    for (f=0; f<n; f++) {
      PetscInt k = element->ifix[f];
      PetscInt a = k / element->dof;
      PetscInt c = k % element->dof;
      if (element->rowmap[0] == element->colmap[a])
        {
          PetscScalar v = element->vfix[f];
          PetscScalar u = element->ufix[f];
          F[c] = u - v;
        }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IGAElementFixJacobian(IGAElement element,PetscScalar J[])
{
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  PetscValidScalarPointer(J,2);
  if (PetscUnlikely(element->collocation)) goto collocation;
  {
    PetscInt M = element->neq * element->dof;
    PetscInt N = element->nen * element->dof;
    PetscInt f,n;
    n = element->nfix;
    for (f=0; f<n; f++) {
      PetscInt i,j,k=element->ifix[f];
      for (i=0; i<M; i++) J[i*N+k] = 0.0;
      for (j=0; j<N; j++) J[k*N+j] = 0.0;
      J[k*N+k] = 1.0;
    }
  }
  PetscFunctionReturn(0);
 collocation:
  {
    PetscInt nen = element->nen;
    PetscInt dof = element->dof;
    PetscInt f,n;
    n = element->nfix;
    for (f=0; f<n; f++) {
      PetscInt k = element->ifix[f];
      PetscInt a = k / dof;
      PetscInt c = k % dof;
      if (element->rowmap[0] == element->colmap[a])
        {
          PetscInt  i,j,N=nen*dof;
          PetscReal *shape = element->basis[0];
          for (j=0; j<N; j++) J[c*N+j] = 0.0;
          for (i=0; i<nen; i++)
            J[c*N+i*dof+c] = shape[i];
        }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IGAElementAssembleVec(IGAElement element,const PetscScalar F[],Vec vec)
{
  PetscInt       mm,*ii;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  PetscValidScalarPointer(F,2);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,3);
  mm = element->neq; ii = element->rowmap;
  if (element->dof == 1) {
    ierr = VecSetValuesLocal(vec,mm,ii,F,ADD_VALUES);CHKERRQ(ierr);
  } else {
    ierr = VecSetValuesBlockedLocal(vec,mm,ii,F,ADD_VALUES);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode IGAElementAssembleMat(IGAElement element,const PetscScalar K[],Mat mat)
{
  PetscInt       mm,*ii;
  PetscInt       nn,*jj;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  PetscValidScalarPointer(K,2);
  PetscValidHeaderSpecific(mat,MAT_CLASSID,3);
  mm = element->neq; ii = element->rowmap;
  nn = element->nen; jj = element->colmap;
  if (element->dof == 1) {
    ierr = MatSetValuesLocal(mat,mm,ii,nn,jj,K,ADD_VALUES);CHKERRQ(ierr);
  } else {
    ierr = MatSetValuesBlockedLocal(mat,mm,ii,nn,jj,K,ADD_VALUES);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
