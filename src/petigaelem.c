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
  size_t MAX_WORK_VAL;
  size_t MAX_WORK_VEC;
  size_t MAX_WORK_MAT;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  MAX_WORK_VAL = sizeof(element->wval)/sizeof(PetscScalar*);
  for (i=0; i<MAX_WORK_VAL; i++)
    {ierr = PetscFree(element->wval[i]);CHKERRQ(ierr);}
  element->nval = 0;
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
#define __FUNCT__ "IGAElementFreeFix"
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

#undef  __FUNCT__
#define __FUNCT__ "IGAElementReset"
PetscErrorCode IGAElementReset(IGAElement element)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!element) PetscFunctionReturn(0);
  PetscValidPointer(element,1);
  element->count =  0;
  element->index = -1;

  ierr = PetscFree(element->mapping);CHKERRQ(ierr);
  ierr = PetscFree(element->geometryX);CHKERRQ(ierr);
  ierr = PetscFree(element->rationalW);CHKERRQ(ierr);
  ierr = PetscFree(element->propertyA);CHKERRQ(ierr);

  ierr = PetscFree(element->weight);CHKERRQ(ierr);
  ierr = PetscFree(element->detJac);CHKERRQ(ierr);

  ierr = PetscFree(element->point);CHKERRQ(ierr);
  ierr = PetscFree(element->scale);CHKERRQ(ierr);
  ierr = PetscFree(element->basis[0]);CHKERRQ(ierr);
  ierr = PetscFree(element->basis[1]);CHKERRQ(ierr);
  ierr = PetscFree(element->basis[2]);CHKERRQ(ierr);
  ierr = PetscFree(element->basis[3]);CHKERRQ(ierr);

  ierr = PetscFree(element->detX);CHKERRQ(ierr);
  ierr = PetscFree(element->gradX[0]);CHKERRQ(ierr);
  ierr = PetscFree(element->gradX[1]);CHKERRQ(ierr);
  ierr = PetscFree(element->shape[0]);CHKERRQ(ierr);
  ierr = PetscFree(element->shape[1]);CHKERRQ(ierr);
  ierr = PetscFree(element->shape[2]);CHKERRQ(ierr);
  ierr = PetscFree(element->shape[3]);CHKERRQ(ierr);

  ierr = IGAElementFreeFix(element);CHKERRQ(ierr);
  ierr = IGAElementFreeWork(element);CHKERRQ(ierr);
  ierr = IGAPointReset(element->iterator);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAElementInit"
PetscErrorCode IGAElementInit(IGAElement element,IGA iga)
{
  PetscInt *start;
  PetscInt *width;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  PetscValidHeaderSpecific(iga,IGA_CLASSID,2);
  IGACheckSetUp(iga,1);
  ierr = IGAElementReset(element);CHKERRQ(ierr);
  element->parent = iga;

  element->dof = iga->dof;
  element->dim = iga->dim;
  element->nsd = iga->geometry ? iga->geometry : iga->dim;
  element->npd = iga->property ? iga->property : 0;
  if (!element->collocation) {
    start = iga->elem_start;
    width = iga->elem_width;
    element->BD = iga->elem_basis;
  } else {
    start = iga->node_lstart;
    width = iga->node_lwidth;
    element->BD = iga->node_basis;
  }
  { /* */
    PetscInt i,dim = element->dim;
    PetscInt nel=1,nqp=1,nen=1;
    for (i=0; i<dim; i++) {
      element->start[i] = start[i];
      element->width[i] = width[i];
      nel *= element->width[i];
      nqp *= element->BD[i]->nqp;
      nen *= element->BD[i]->nen;
    }
    for (i=dim; i<3; i++) {
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
    PetscInt nsd = element->nsd;
    PetscInt npd = element->npd;
    PetscInt nen = element->nen;
    PetscInt nqp = element->nqp;

    ierr = PetscMalloc1(nen,PetscInt,&element->mapping);CHKERRQ(ierr);
    ierr = PetscMalloc1(nen*nsd,PetscReal,&element->geometryX);CHKERRQ(ierr);
    ierr = PetscMalloc1(nen,PetscReal,&element->rationalW);CHKERRQ(ierr);
    ierr = PetscMalloc1(nen*npd,PetscScalar,&element->propertyA);CHKERRQ(ierr);

    ierr = PetscMalloc1(nqp,PetscReal,&element->weight);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp,PetscReal,&element->detJac);CHKERRQ(ierr);

    ierr = PetscMalloc1(nqp*dim,PetscReal,&element->point);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*dim,PetscReal,&element->scale);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*nen,PetscReal,&element->basis[0]);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*nen*dim,PetscReal,&element->basis[1]);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*nen*dim*dim,PetscReal,&element->basis[2]);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*nen*dim*dim*dim,PetscReal,&element->basis[3]);CHKERRQ(ierr);

    ierr = PetscMalloc1(nqp,PetscReal,&element->detX);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*dim*dim,PetscReal,&element->gradX[0]);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*dim*dim,PetscReal,&element->gradX[1]);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*nen,PetscReal,&element->shape[0]);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*nen*dim,PetscReal,&element->shape[1]);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*nen*dim*dim,PetscReal,&element->shape[2]);CHKERRQ(ierr);
    ierr = PetscMalloc1(nqp*nen*dim*dim*dim,PetscReal,&element->shape[3]);CHKERRQ(ierr);

    ierr = PetscMemzero(element->weight,  sizeof(PetscReal)*nqp);CHKERRQ(ierr);
    ierr = PetscMemzero(element->detJac,  sizeof(PetscReal)*nqp);CHKERRQ(ierr);

    ierr = PetscMemzero(element->point,   sizeof(PetscReal)*nqp*dim);CHKERRQ(ierr);
    ierr = PetscMemzero(element->scale,   sizeof(PetscReal)*nqp*dim);CHKERRQ(ierr);
    ierr = PetscMemzero(element->basis[0],sizeof(PetscReal)*nqp*nen);CHKERRQ(ierr);
    ierr = PetscMemzero(element->basis[1],sizeof(PetscReal)*nqp*nen*dim);CHKERRQ(ierr);
    ierr = PetscMemzero(element->basis[2],sizeof(PetscReal)*nqp*nen*dim*dim);CHKERRQ(ierr);
    ierr = PetscMemzero(element->basis[3],sizeof(PetscReal)*nqp*nen*dim*dim*dim);CHKERRQ(ierr);

    ierr = PetscMemzero(element->detX,    sizeof(PetscReal)*nqp);CHKERRQ(ierr);
    ierr = PetscMemzero(element->gradX[0],sizeof(PetscReal)*nqp*dim*dim);CHKERRQ(ierr);
    ierr = PetscMemzero(element->gradX[1],sizeof(PetscReal)*nqp*dim*dim);CHKERRQ(ierr);
    ierr = PetscMemzero(element->shape[0],sizeof(PetscReal)*nqp*nen);CHKERRQ(ierr);
    ierr = PetscMemzero(element->shape[1],sizeof(PetscReal)*nqp*nen*dim);CHKERRQ(ierr);
    ierr = PetscMemzero(element->shape[2],sizeof(PetscReal)*nqp*nen*dim*dim);CHKERRQ(ierr);
    ierr = PetscMemzero(element->shape[3],sizeof(PetscReal)*nqp*nen*dim*dim*dim);CHKERRQ(ierr);
  }
  if (!element->collocation) {
    element->neq    = element->nen;
    element->rowmap = element->mapping;
    element->colmap = element->mapping;
  } else {
    element->neq    = 1;
    element->rowmap = &element->index;
    element->colmap = element->mapping;
  }
  { /* */
    PetscInt nen = element->nen;
    PetscInt dof = element->dof;
    element->nfix = 0;
    ierr = PetscMalloc1(nen*dof,PetscInt,   &element->ifix);CHKERRQ(ierr);
    ierr = PetscMalloc1(nen*dof,PetscScalar,&element->vfix);CHKERRQ(ierr);
    ierr = PetscMalloc1(nen*dof,PetscScalar,&element->ufix);CHKERRQ(ierr);
    element->nflux = 0;
    ierr = PetscMalloc1(nen*dof,PetscInt,   &element->iflux);CHKERRQ(ierr);
    ierr = PetscMalloc1(nen*dof,PetscScalar,&element->vflux);CHKERRQ(ierr);
  }
  ierr = IGAPointInit(element->iterator,element);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAGetElement"
PetscErrorCode IGAGetElement(IGA iga,IGAElement *element)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(element,2);
  if (!iga->collocation)
    *element = iga->elem_iterator;
  else
    *element = iga->node_iterator;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGABeginElement"
PetscErrorCode IGABeginElement(IGA iga,IGAElement *element)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(element,2);
  IGACheckSetUp(iga,1);
  ierr = IGAGetElement(iga,element);CHKERRQ(ierr);
  (*element)->index = -1;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGANextElement"
PetscBool IGANextElement(IGA iga,IGAElement element)
{
  PetscInt i,dim  = element->dim;
  PetscInt *start = element->start;
  PetscInt *width = element->width;
  PetscInt *ID    = element->ID;
  PetscInt index,coord;
  PetscErrorCode ierr;
  /* */
  element->nval = 0;
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
    coord = index % width[i];
    index = (index - coord) / width[i];
    ID[i] = coord + start[i];
  }
#undef  CHKERRRETURN
#define CHKERRRETURN(n,r) do{if(PetscUnlikely(n)){CHKERRCONTINUE(n);return(r);}}while(0)
  ierr = IGAElementBuildMapping(element);  CHKERRRETURN(ierr,PETSC_FALSE);
  ierr = IGAElementBuildGeometry(element); CHKERRRETURN(ierr,PETSC_FALSE);
  ierr = IGAElementBuildProperty(element); CHKERRRETURN(ierr,PETSC_FALSE);
  ierr = IGAElementBuildFix(element);      CHKERRRETURN(ierr,PETSC_FALSE);
#undef  CHKERRRETURN
  return PETSC_TRUE;
}

#undef  __FUNCT__
#define __FUNCT__ "IGANextBoundaryElement"
PetscBool IGANextBoundaryElement(IGA iga,IGAElement element,PetscInt dir,PetscInt side)
{
  PetscInt i,dim  = element->dim;
  PetscInt *start = element->start;
  PetscInt *width = element->width;
  PetscInt *ID    = element->ID;
  PetscInt nel    = iga->axis[dir]->nel;
  PetscInt index,coord;
  PetscErrorCode ierr;
  /* */
  element->nval = 0;
  element->nvec = 0;
  element->nmat = 0;
  /* */

  do {
    index = ++element->index;
    if (PetscUnlikely(index >= element->count)) {
      element->index = -1;
      return PETSC_FALSE;
    }
    /* */
    for (i=0; i<dim; i++) {
      coord = index % width[i];
      index = (index - coord) / width[i];
      ID[i] = coord + start[i];
    }
  }while( ID[dir] != side*(nel-1) );

#undef  CHKERRRETURN
#define CHKERRRETURN(n,r) do{if(PetscUnlikely(n)){CHKERRCONTINUE(n);return(r);}}while(0)
  ierr = IGAElementBuildMapping(element);  CHKERRRETURN(ierr,PETSC_FALSE);
  ierr = IGAElementBuildGeometry(element); CHKERRRETURN(ierr,PETSC_FALSE);
  ierr = IGAElementBuildProperty(element); CHKERRRETURN(ierr,PETSC_FALSE);
  ierr = IGAElementBuildFix(element);      CHKERRRETURN(ierr,PETSC_FALSE);
#undef  CHKERRRETURN
  return PETSC_TRUE;
}

#undef  __FUNCT__
#define __FUNCT__ "IGAEndElement"
PetscErrorCode IGAEndElement(IGA iga,IGAElement *element)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(element,2);
  PetscValidPointer(*element,2);
  if (PetscUnlikely((*element)->index != -1)) {
    (*element)->index = -1;
    *element = PETSC_NULL;
    PetscFunctionReturn(PETSC_ERR_PLIB);
  }
  *element = PETSC_NULL;
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

#undef  __FUNCT__
#define __FUNCT__ "IGAElementBeginPoint"
PetscErrorCode IGAElementBeginPoint(IGAElement element,IGAPoint *point)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  PetscValidPointer(point,2);
  if (PetscUnlikely(element->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during element loop");
  ierr = IGAElementBuildQuadrature(element);CHKERRQ(ierr);
  ierr = IGAElementBuildShapeFuns(element);CHKERRQ(ierr);
  *point = element->iterator;

  (*point)->count = element->nqp;
  (*point)->index = -1;

  (*point)->neq = element->neq;
  (*point)->nen = element->nen;
  (*point)->dof = element->dof;
  (*point)->dim = element->dim;
  (*point)->nsd = element->nsd;
  (*point)->npd = element->npd;

  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "IGABoundaryElementBeginPoint"
PetscErrorCode IGABoundaryElementBeginPoint(IGAElement element,IGAPoint *point,PetscInt dir,PetscInt side)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  PetscValidPointer(point,2);
  if (PetscUnlikely(element->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during element loop");
  ierr = IGAElementBuildBoundaryQuadrature(element,dir,side);CHKERRQ(ierr);
  ierr = IGAElementBuildBoundaryShapeFuns(element,dir,side);CHKERRQ(ierr);
  *point = element->iterator;

  (*point)->count = element->nqp;
  (*point)->index = -1;

  (*point)->neq = element->neq;
  (*point)->nen = element->nen;
  (*point)->dof = element->dof;
  (*point)->dim = element->dim;
  (*point)->nsd = element->nsd;
  (*point)->npd = element->npd;

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAElementNextPoint"
PetscBool IGAElementNextPoint(IGAElement element,IGAPoint point)
{
  PetscInt nen = point->nen;
  PetscInt dim = point->dim;
  PetscInt nsd = point->nsd;
  PetscInt index;
  /* */
  point->nvec = 0;
  point->nmat = 0;
  /* */
  index = ++point->index;
  if (PetscUnlikely(index == 0))            goto start;
  if (PetscUnlikely(index >= point->count)) goto stop;

  point->weight   += 1;
  point->detJac   += 1;

  point->point    += dim;
  point->scale    += dim;
  point->basis[0] += nen;
  point->basis[1] += nen*dim;
  point->basis[2] += nen*dim*dim;
  point->basis[3] += nen*dim*dim*dim;

  point->detX     += 1;
  point->gradX[0] += dim*dim;
  point->gradX[1] += dim*dim;
  point->shape[0] += nen;
  point->shape[1] += nen*dim;
  point->shape[2] += nen*dim*dim;
  point->shape[3] += nen*dim*dim*dim;

  return PETSC_TRUE;

 start:

  point->geometry = element->geometryX;
  point->property = element->propertyA;
  if (!element->geometry)
    point->geometry = PETSC_NULL;
  if (!element->property)
    point->property = PETSC_NULL;

  point->weight   = element->weight;
  point->detJac   = element->detJac;

  point->point    = element->point;
  point->scale    = element->scale;
  point->basis[0] = element->basis[0];
  point->basis[1] = element->basis[1];
  point->basis[2] = element->basis[2];
  point->basis[3] = element->basis[3];

  if (element->geometry && dim == nsd) { /* XXX */
    point->detX     = element->detX;
    point->gradX[0] = element->gradX[0];
    point->gradX[1] = element->gradX[1];
    point->shape[0] = element->shape[0];
    point->shape[1] = element->shape[1];
    point->shape[2] = element->shape[2];
    point->shape[3] = element->shape[3];
  } else {
    point->shape[0] = element->basis[0];
    point->shape[1] = element->basis[1];
    point->shape[2] = element->basis[2];
    point->shape[3] = element->basis[3];
  }

  return PETSC_TRUE;

 stop:

  point->index = -1;
  return PETSC_FALSE;

}

#undef  __FUNCT__
#define __FUNCT__ "IGABoundaryElementNextPoint"
PetscBool IGABoundaryElementNextPoint(IGAElement element,IGAPoint point,PetscInt dir,PetscInt side)
{
  PetscInt nen = point->nen;
  PetscInt dim = point->dim;
  PetscInt nsd = point->nsd;
  PetscInt index,skip=1,i;
  /* */
  point->nvec = 0;
  point->nmat = 0;
  /* */
  for(i=0;i<dim;i++) if(i != dir) skip *= element->parent->rule[i]->nqp;

  do {
    index = ++point->index;
    if (PetscUnlikely(index == 0))            goto start;
    if (PetscUnlikely(index >= point->count)) goto stop;

    point->weight   += 1;
    point->detJac   += 1;

    point->point    += dim;
    point->scale    += dim;
    point->basis[0] += nen;
    point->basis[1] += nen*dim;
    point->basis[2] += nen*dim*dim;
    point->basis[3] += nen*dim*dim*dim;

    point->detX     += 1;
    point->gradX[0] += dim*dim;
    point->gradX[1] += dim*dim;
    point->shape[0] += nen;
    point->shape[1] += nen*dim;
    point->shape[2] += nen*dim*dim;
    point->shape[3] += nen*dim*dim*dim;
  } while( index % skip != 0 );

  return PETSC_TRUE;

 start:

  point->geometry = element->geometryX;
  point->property = element->propertyA;
  if (!element->geometry)
    point->geometry = PETSC_NULL;
  if (!element->property)
    point->property = PETSC_NULL;

  point->weight   = element->weight;
  point->detJac   = element->detJac;

  point->point    = element->point;
  point->scale    = element->scale;
  point->basis[0] = element->basis[0];
  point->basis[1] = element->basis[1];
  point->basis[2] = element->basis[2];
  point->basis[3] = element->basis[3];

  if (element->geometry && dim == nsd) { /* XXX */
    point->detX     = element->detX;
    point->gradX[0] = element->gradX[0];
    point->gradX[1] = element->gradX[1];
    point->shape[0] = element->shape[0];
    point->shape[1] = element->shape[1];
    point->shape[2] = element->shape[2];
    point->shape[3] = element->shape[3];
  } else {
    point->shape[0] = element->basis[0];
    point->shape[1] = element->basis[1];
    point->shape[2] = element->basis[2];
    point->shape[3] = element->basis[3];
  }

  return PETSC_TRUE;

 stop:

  point->index = -1;
  return PETSC_FALSE;

}

#undef  __FUNCT__
#define __FUNCT__ "IGAElementEndPoint"
PetscErrorCode IGAElementEndPoint(IGAElement element,IGAPoint *point)
{
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  PetscValidPointer(point,2);
  PetscValidPointer(*point,2);
  if (PetscUnlikely((*point)->index != -1)) {
    (*point)->index = -1;
    PetscFunctionReturn(PETSC_ERR_PLIB);
  }
  *point = PETSC_NULL;
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
#define __FUNCT__ "IGAElementGetCount"
PetscErrorCode IGAElementGetCount(IGAElement element,PetscInt *count)
{
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  PetscValidIntPointer(count,2);
  *count = element->count;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAElementGetSizes"
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

#undef  __FUNCT__
#define __FUNCT__ "IGAElementGetMapping"
PetscErrorCode IGAElementGetMapping(IGAElement element,PetscInt *nen,const PetscInt *mapping[])
{
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  if (nen)     PetscValidIntPointer(nen,3);
  if (mapping) PetscValidPointer(mapping,3);
  if (nen)     *nen     = element->nen;
  if (mapping) *mapping = element->mapping;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAElementGetQuadrature"
PetscErrorCode IGAElementGetQuadrature(IGAElement element,PetscInt *nqp,PetscInt *dim,
				       const PetscReal *point[],const PetscReal *weight[],
				       const PetscReal *detJac[])
{
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  if (nqp)    PetscValidIntPointer(nqp,2);
  if (dim)    PetscValidIntPointer(dim,3);
  if (point)  PetscValidPointer(point,4);
  if (weight) PetscValidPointer(weight,5);
  if (detJac) PetscValidPointer(detJac,6);
  if (nqp)    *nqp    = element->nqp;
  if (dim)    *dim    = element->dim;
  if (point)  *point  = element->point;
  if (weight) *weight = element->weight;
  if (detJac) *detJac = element->detJac;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAElementGetShapeFuns"
PetscErrorCode IGAElementGetShapeFuns(IGAElement element,PetscInt *nqp,
				      PetscInt *nen,PetscInt *dim,
				      const PetscReal **shapefuns[])
{
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  if (nqp)       PetscValidIntPointer(nqp,2);
  if (nen)       PetscValidIntPointer(nen,3);
  if (dim)       PetscValidIntPointer(dim,4);
  if (shapefuns) PetscValidPointer(shapefuns,5);
  if (nqp)       *nqp       = element->nqp;
  if (nen)       *nqp       = element->nqp;
  if (dim)       *dim       = element->dim;
  if (shapefuns) *shapefuns = (const PetscReal **)element->shape;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

#undef  __FUNCT__
#define __FUNCT__ "IGAElementBuildMapping"
PetscErrorCode IGAElementBuildMapping(IGAElement element)
{
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  if (PetscUnlikely(element->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during element loop");
  { /* */
    IGA      iga = element->parent;
    IGABasis *BD = element->BD;
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
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAElementBuildGeometry"
PetscErrorCode IGAElementBuildGeometry(IGAElement element)
{
  IGA iga;
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  if (PetscUnlikely(element->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during element loop");
  iga = element->parent;
  element->geometry = iga->geometry ? PETSC_TRUE : PETSC_FALSE;
  element->rational = iga->rational ? PETSC_TRUE : PETSC_FALSE;
  if (element->geometry || element->rational) {
    const PetscInt  *map = element->mapping;
    const PetscReal *arrayX = iga->geometryX;
    const PetscReal *arrayW = iga->rationalW;
    PetscReal *X = element->geometryX;
    PetscReal *W = element->rationalW;
    PetscInt a,nen = element->nen;
    PetscInt i,nsd = element->nsd;
    if (element->geometry)
      for (a=0; a<nen; a++)
	for (i=0; i<nsd; i++)
	  X[i + a*nsd] = arrayX[i + map[a]*nsd];
    if (element->rational)
      for (a=0; a<nen; a++)
	W[a] = arrayW[map[a]];
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAElementBuildProperty"
PetscErrorCode IGAElementBuildProperty(IGAElement element)
{
  IGA iga;
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  if (PetscUnlikely(element->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during element loop");
  iga = element->parent;
  element->property = iga->property ? PETSC_TRUE: PETSC_FALSE;
  if (element->property) {
    const PetscInt *map = element->mapping;
    const PetscScalar *arrayA = iga->propertyA;
    PetscScalar *A = element->propertyA;
    PetscInt a,nen = element->nen;
    PetscInt i,npd = element->npd;
    for (a=0; a<nen; a++)
      for (i=0; i<npd; i++)
	A[i + a*npd] = arrayA[i + map[a]*npd];
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
extern void IGA_Quadrature_1D(PetscInt,const PetscReal[],const PetscReal[],const PetscReal*,
			      PetscReal[],PetscReal[],PetscReal*,PetscReal[]);
extern void IGA_Quadrature_2D(PetscInt,const PetscReal[],const PetscReal[],const PetscReal*,
			      PetscInt,const PetscReal[],const PetscReal[],const PetscReal*,
			      PetscReal[],PetscReal[],PetscReal*,PetscReal[]);
extern void IGA_Quadrature_3D(PetscInt,const PetscReal[],const PetscReal[],const PetscReal*,
			      PetscInt,const PetscReal[],const PetscReal[],const PetscReal*,
			      PetscInt,const PetscReal[],const PetscReal[],const PetscReal*,
			      PetscReal[],PetscReal[],PetscReal*,PetscReal[]);
EXTERN_C_END

#define IGA_Quadrature_ARGS(ID,BD,i) \
  BD[i]->nqp,BD[i]->point+ID[i]*BD[i]->nqp,BD[i]->weight,BD[i]->detJ+ID[i]

#undef  __FUNCT__
#define __FUNCT__ "IGAElementBuildQuadrature"
PetscErrorCode IGAElementBuildQuadrature(IGAElement element)
{
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  if (PetscUnlikely(element->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during element loop");
  {
    IGABasis *BD = element->BD;
    PetscInt *ID = element->ID;
    switch (element->dim) {
    case 3: IGA_Quadrature_3D(IGA_Quadrature_ARGS(ID,BD,0),
			      IGA_Quadrature_ARGS(ID,BD,1),
			      IGA_Quadrature_ARGS(ID,BD,2),
			      element->weight,element->detJac,
			      element->point,element->scale); break;
    case 2: IGA_Quadrature_2D(IGA_Quadrature_ARGS(ID,BD,0),
			      IGA_Quadrature_ARGS(ID,BD,1),
			      element->weight,element->detJac,
			      element->point,element->scale); break;
    case 1: IGA_Quadrature_1D(IGA_Quadrature_ARGS(ID,BD,0),
			      element->weight,element->detJac,
			      element->point,element->scale); break;
    }
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAElementBuildBoundaryQuadrature"
PetscErrorCode IGAElementBuildBoundaryQuadrature(IGAElement element,PetscInt dir,PetscInt side)
{
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  if (PetscUnlikely(element->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during element loop");
  {
    IGABasis   *BD = element->BD;
    PetscInt i,*ID = element->ID;

    /* Sad hack */
    PetscScalar pts[BD[dir]->nqp],wts[BD[dir]->nqp];
    PetscReal U[2] = {0,0}, one=1;
    IGAAxisGetLimits(element->parent->axis[dir],&U[0],&U[1]);
    for(i=0;i<BD[dir]->nqp;i++){
      pts[i] = U[side];
      wts[i] = 1.0;
    }

    switch (element->dim) {
    case 3:
      if(dir==0){
	IGA_Quadrature_3D(BD[0]->nqp,&pts[0],&wts[0],&one,
			  IGA_Quadrature_ARGS(ID,BD,1),
			  IGA_Quadrature_ARGS(ID,BD,2),
			  element->weight,element->detJac,
			  element->point,element->scale);
      }else if(dir==1){
	IGA_Quadrature_3D(IGA_Quadrature_ARGS(ID,BD,0),
			  BD[1]->nqp,&pts[0],&wts[0],&one,
			  IGA_Quadrature_ARGS(ID,BD,2),
			  element->weight,element->detJac,
			  element->point,element->scale);
      }else{
	IGA_Quadrature_3D(IGA_Quadrature_ARGS(ID,BD,0),
			  IGA_Quadrature_ARGS(ID,BD,1),
			  BD[2]->nqp,&pts[0],&wts[0],&one,
			  element->weight,element->detJac,
			  element->point,element->scale);
      }
      break;
    case 2:
      if(dir==0){
	IGA_Quadrature_2D(BD[0]->nqp,&pts[0],&wts[0],&one,
			  IGA_Quadrature_ARGS(ID,BD,1),
			  element->weight,element->detJac,
			  element->point,element->scale);
      }else{
	IGA_Quadrature_2D(IGA_Quadrature_ARGS(ID,BD,0),
			  BD[1]->nqp,&pts[0],&wts[0],&one,
			  element->weight,element->detJac,
			  element->point,element->scale);
      }
      break;
    case 1: IGA_Quadrature_1D(IGA_Quadrature_ARGS(ID,BD,0),
			      element->weight,element->detJac,
			      element->point,element->scale); break;
    }
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

#define IGA_BasisFuns_ARGS(ID,BD,i) \
  BD[i]->nqp,BD[i]->nen,BD[i]->d,   \
  BD[i]->value+ID[i]*BD[i]->nqp*BD[i]->nen*(BD[i]->d+1)

#undef  __FUNCT__
#define __FUNCT__ "IGAElementBuildShapeFuns"
PetscErrorCode IGAElementBuildShapeFuns(IGAElement element)
{
  PetscInt order;
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  if (PetscUnlikely(element->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during element loop");
  order = element->parent->order;
  {
    IGABasis *BD  = element->BD;
    PetscInt *ID  = element->ID;
    PetscReal **N = element->basis;
    switch (element->dim) {
    case 3: IGA_BasisFuns_3D(order,element->rational,
			     element->rationalW,
			     IGA_BasisFuns_ARGS(ID,BD,0),
			     IGA_BasisFuns_ARGS(ID,BD,1),
			     IGA_BasisFuns_ARGS(ID,BD,2),
			     N[0],N[1],N[2],N[3]); break;
    case 2: IGA_BasisFuns_2D(order,element->rational,
			     element->rationalW,
			     IGA_BasisFuns_ARGS(ID,BD,0),
			     IGA_BasisFuns_ARGS(ID,BD,1),
			     N[0],N[1],N[2],N[3]); break;
    case 1: IGA_BasisFuns_1D(order,element->rational,
			     element->rationalW,
			     IGA_BasisFuns_ARGS(ID,BD,0),
			     N[0],N[1],N[2],N[3]); break;
    }
  }
  if (element->dim == element->nsd) /* XXX */
  if (element->geometry) {
    PetscInt q, nqp = element->nqp;
    PetscReal **M = element->basis;
    PetscReal **N = element->shape;
    PetscReal *dX = element->detX;
    PetscReal **gX = element->gradX;
    switch (element->dim) {
    case 3: IGA_ShapeFuns_3D(order,
			     element->nqp,element->nen,
			     element->geometryX,
			     M[0],M[1],M[2],M[3],
			     N[0],N[1],N[2],N[3],
			     dX,gX[0],gX[1]); break;
    case 2: IGA_ShapeFuns_2D(order,
			     element->nqp,element->nen,
			     element->geometryX,
			     M[0],M[1],M[2],M[3],
			     N[0],N[1],N[2],N[3],
			     dX,gX[0],gX[1]); break;
    case 1: IGA_ShapeFuns_1D(order,
			     element->nqp,element->nen,
			     element->geometryX,
			     M[0],M[1],M[2],M[3],
			     N[0],N[1],N[2],N[3],
			     dX,gX[0],gX[1]); break;
    }
    for (q=0; q<nqp; q++)
      element->detJac[q] *= dX[q];
  }
  PetscFunctionReturn(0);
}

extern void IGA_DersBasisFuns(PetscInt i,PetscReal u,PetscInt p,PetscInt d,const PetscReal U[],PetscReal N[]);

#undef  __FUNCT__
#define __FUNCT__ "IGAElementBuildBoundaryShapeFuns"
PetscErrorCode IGAElementBuildBoundaryShapeFuns(IGAElement element,PetscInt dir,PetscInt side)
{
  PetscInt order;
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  if (PetscUnlikely(element->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during element loop");
  order = element->parent->order;
  {
    IGABasis *BD  = element->BD;
    PetscInt *ID  = element->ID;
    PetscReal **N = element->basis;

    PetscInt k = element->parent->axis[dir]->span[element->ID[dir]];
    PetscReal u = element->point[dir];
    PetscInt p = element->parent->axis[dir]->p;
    PetscInt d = element->parent->order;
    PetscReal *U = element->parent->axis[dir]->U;
    PetscInt nqp = BD[dir]->nqp;
    PetscInt nen = BD[dir]->nen;
    PetscReal value[nqp*nen*(d+1)];
    IGA_DersBasisFuns(k,u,p,d,U,&value[0]);

    switch (element->dim) {
    case 3:
      if(dir == 0){
	IGA_BasisFuns_3D(order,element->rational,
			 element->rationalW,
			 BD[0]->nqp,BD[0]->nen,BD[0]->d,&value[0],
			 IGA_BasisFuns_ARGS(ID,BD,1),
			 IGA_BasisFuns_ARGS(ID,BD,2),
			 N[0],N[1],N[2],N[3]);
      }else if(dir == 1){
	IGA_BasisFuns_3D(order,element->rational,
			 element->rationalW,
			 IGA_BasisFuns_ARGS(ID,BD,0),
			 BD[1]->nqp,BD[1]->nen,BD[1]->d,&value[0],
			 IGA_BasisFuns_ARGS(ID,BD,2),
			 N[0],N[1],N[2],N[3]);
      }else{
	IGA_BasisFuns_3D(order,element->rational,
			 element->rationalW,
			 IGA_BasisFuns_ARGS(ID,BD,0),
			 IGA_BasisFuns_ARGS(ID,BD,1),
			 BD[2]->nqp,BD[2]->nen,BD[2]->d,&value[0],
			 N[0],N[1],N[2],N[3]);
      }
      break;
    case 2:
      if(dir == 0){
	IGA_BasisFuns_2D(order,element->rational,
			 element->rationalW,
			 BD[0]->nqp,BD[0]->nen,BD[0]->d,&value[0],
			 IGA_BasisFuns_ARGS(ID,BD,1),
			 N[0],N[1],N[2],N[3]);
      }else{
	IGA_BasisFuns_2D(order,element->rational,
			 element->rationalW,
			 IGA_BasisFuns_ARGS(ID,BD,0),
			 BD[1]->nqp,BD[1]->nen,BD[1]->d,&value[0],
			 N[0],N[1],N[2],N[3]);
      }
      break;
    case 1: IGA_BasisFuns_1D(order,element->rational,
			     element->rationalW,
			     IGA_BasisFuns_ARGS(ID,BD,0),
			     N[0],N[1],N[2],N[3]); break;
    }
  }
  if (element->dim == element->nsd) /* XXX */
  if (element->geometry) {
    PetscInt q, nqp = element->nqp;
    PetscReal **M = element->basis;
    PetscReal **N = element->shape;
    PetscReal *dX = element->detX;
    PetscReal **gX = element->gradX;
    switch (element->dim) {
    case 3: IGA_ShapeFuns_3D(order,
			     element->nqp,element->nen,
			     element->geometryX,
			     M[0],M[1],M[2],M[3],
			     N[0],N[1],N[2],N[3],
			     dX,gX[0],gX[1]); break;
    case 2: IGA_ShapeFuns_2D(order,
			     element->nqp,element->nen,
			     element->geometryX,
			     M[0],M[1],M[2],M[3],
			     N[0],N[1],N[2],N[3],
			     dX,gX[0],gX[1]); break;
    case 1: IGA_ShapeFuns_1D(order,
			     element->nqp,element->nen,
			     element->geometryX,
			     M[0],M[1],M[2],M[3],
			     N[0],N[1],N[2],N[3],
			     dX,gX[0],gX[1]); break;
    }
    for (q=0; q<nqp; q++)
      element->detJac[q] *= dX[q];
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

#undef  __FUNCT__
#define __FUNCT__ "IGAElementGetWorkVal"
PetscErrorCode IGAElementGetWorkVal(IGAElement element,PetscScalar *U[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  PetscValidPointer(U,2);
  if (PetscUnlikely(element->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during element loop");
  {
    size_t MAX_WORK_VAL = sizeof(element->wval)/sizeof(PetscScalar*);
    PetscInt n = element->nen * element->dof;
    if (PetscUnlikely(element->nval >= (PetscInt)MAX_WORK_VAL))
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many work values requested");
    if (PetscUnlikely(!element->wval[element->nval])) {
      ierr = PetscMalloc1(n,PetscScalar,&element->wval[element->nval]);CHKERRQ(ierr);
    }
    *U = element->wval[element->nval++];
    ierr = PetscMemzero(*U,n*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

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
    PetscInt m = element->neq * element->dof;
    PetscInt n = element->nen * element->dof;
    if (PetscUnlikely(element->nvec >= (PetscInt)MAX_WORK_VEC))
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many work vectors requested");
    if (PetscUnlikely(!element->wvec[element->nvec])) {
      ierr = PetscMalloc1(n,PetscScalar,&element->wvec[element->nvec]);CHKERRQ(ierr);
    }
    *V = element->wvec[element->nvec++];
    ierr = PetscMemzero(*V,m*sizeof(PetscScalar));CHKERRQ(ierr);
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
    PetscInt m = element->neq * element->dof;
    PetscInt n = element->nen * element->dof;
    if (PetscUnlikely(element->nmat >= (PetscInt)MAX_WORK_MAT))
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many work matrices requested");
    if (PetscUnlikely(!element->wmat[element->nmat])) {
      ierr = PetscMalloc1(n*n,PetscScalar,&element->wmat[element->nmat]);CHKERRQ(ierr);
    }
    *M = element->wmat[element->nmat++];
    ierr = PetscMemzero(*M,m*n*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

#undef  __FUNCT__
#define __FUNCT__ "IGAElementGetValues"
PetscErrorCode IGAElementGetValues(IGAElement element,const PetscScalar arrayU[],PetscScalar U[])
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

/* -------------------------------------------------------------------------- */

static void AddFixa(IGAElement element,IGABoundary b,PetscInt a)
{
  if (b->count) {
    PetscInt dof = element->dof;
    PetscInt count = element->nfix;
    PetscInt *index = element->ifix;
    PetscScalar *value = element->vfix;
    PetscInt j,k,n = b->count;
    for (k=0; k<n; k++) {
      PetscInt idx = a*dof + b->field[k];
      PetscScalar val = b->value[k];
      for (j=0; j<count; j++)
	if (index[j] == idx) break;
      if (j == count) count++;
      index[j] = idx;
      value[j] = val;
    }
    element->nfix = count;
  }
}

EXTERN_C_BEGIN
extern void IGA_BoundaryArea_2D(const PetscInt[],PetscInt,PetscInt,
				PetscInt,const PetscReal[],
				PetscInt,const PetscReal[],
				PetscInt,const PetscReal[],PetscInt,PetscInt,const PetscReal[],
				PetscReal*);
extern void IGA_BoundaryArea_3D(const PetscInt[],PetscInt,PetscInt,
				PetscInt,const PetscReal[],
				PetscInt,const PetscReal[],
				PetscInt,const PetscReal[],PetscInt,PetscInt,const PetscReal[],
				PetscInt,const PetscReal[],PetscInt,PetscInt,const PetscReal[],
				PetscReal*);
EXTERN_C_END

static PetscReal BoundaryArea(IGAElement element,PetscInt dir,PetscInt side)
{
  PetscReal A = 1.0;
  PetscInt *ID = element->ID;
  IGABasis *BD = element->BD;
  PetscInt i,dim = element->dim;
  if (dim == 1) return A;
  for (i=0; i<dim; i++)
    if (i != dir) {
      PetscReal L = BD[i]->detJ[ID[i]];
      PetscInt  n = BD[i]->nen;
      A *= L/n;
    }
  if (!element->geometry) {
    A *= (dim==2) ? 2 : 4; /* sum(W) = 2 */
  } else {
    PetscInt shape[3] = {1,1,1};
    PetscInt k,nqp[3],nen[3],ndr[3];
    PetscReal *W[3],*N[3],dS = 1.0;
    for (i=0; i<dim; i++)
      shape[i] = BD[i]->nen;
    for (k=0,i=0; i<dim; i++) {
      if (i == dir) continue;
      nqp[k] = BD[i]->nqp;
      nen[k] = BD[i]->nen;
      ndr[k] = BD[i]->d;
      W[k]   = BD[i]->weight;
      N[k]   = BD[i]->value+ID[i]*nqp[k]*nen[k]*(ndr[k]+1);
      k++;
    }
    switch (dim) {
    case 2: IGA_BoundaryArea_2D(shape,dir,side,
				element->geometry,element->geometryX,
				element->rational,element->rationalW,
				nqp[0],W[0],nen[0],ndr[0],N[0],
				&dS); break;
    case 3: IGA_BoundaryArea_3D(shape,dir,side,
				element->geometry,element->geometryX,
				element->rational,element->rationalW,
				nqp[0],W[0],nen[0],ndr[0],N[0],
				nqp[1],W[1],nen[1],ndr[1],N[1],
				&dS);break;
    }
    A *= dS;
  }
  return A;
}

static void AddFlux(IGAElement element,IGABoundary b,PetscInt a,PetscReal A)
{
  if (b->nload) {
    PetscInt dof = element->dof;
    PetscInt count = element->nflux;
    PetscInt *index = element->iflux;
    PetscScalar *value = element->vflux;
    PetscInt j,k,n = b->nload;
    for (k=0; k<n; k++) {
      PetscInt idx = a*dof + b->iload[k];
      PetscScalar val = b->vload[k];
      for (j=0; j<count; j++)
	if (index[j] == idx) break;
      if (j == count) value[count++] = 0.0;
      index[j] = idx;
      value[j]+= val*A;
    }
    element->nflux = count;
  }
}

static void BuildFix(IGAElement element,PetscInt dir,PetscInt side)
{
  IGABoundary b = element->parent->boundary[dir][side];
  if (b->count || b->nload) {
    PetscReal Area = b->nload ? BoundaryArea(element,dir,side) : 1.0;
    IGABasis *BD = element->BD;
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
	    AddFixa(element,b,a);
	    AddFlux(element,b,a,Area);
	  }
  }
}

#undef  __FUNCT__
#define __FUNCT__ "IGAElementBuildFix"
PetscErrorCode IGAElementBuildFix(IGAElement element)
{
  PetscFunctionBegin;
  PetscValidPointer(element,1);
  if (PetscUnlikely(element->index < 0))
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call during element loop");
  element->nfix  = 0;
  element->nflux = 0;
  if (!element->collocation) {
    IGAAxis  *AX = element->parent->axis;
    IGABasis *BD = element->BD;
    PetscInt *ID = element->ID;
    PetscInt i,dim = element->dim;
    for (i=0; i<dim; i++) {
      PetscBool w = AX[i]->periodic;
      PetscInt  e = BD[i]->nel-1; /* last element */
      if (ID[i] == 0 && !w) BuildFix(element,i,0);
      if (ID[i] == e && !w) BuildFix(element,i,1);
    }
  } else {
    PetscInt A0[3] = {PETSC_MIN_INT,PETSC_MIN_INT,PETSC_MIN_INT};
    PetscInt A1[3] = {PETSC_MAX_INT,PETSC_MAX_INT,PETSC_MAX_INT};
    {
      IGAAxis  *AX = element->parent->axis;
      PetscInt i,dim = element->dim;
      for (i=0; i<dim; i++) {
	PetscBool w = AX[i]->periodic;
	PetscInt  n = AX[i]->nnp-1; /* last node */
	A0[i] = 0; if (!w) A1[i] = n;
      }
    }
    {
      IGABoundary (*b)[2] = element->parent->boundary;
      IGABasis *BD = element->BD;
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
	      /**/ if (iA == A0[0]) AddFixa(element,b[0][0],a);
	      else if (iA == A1[0]) AddFixa(element,b[0][1],a);
	      /**/ if (jA == A0[1]) AddFixa(element,b[1][0],a);
	      else if (jA == A1[1]) AddFixa(element,b[1][1],a);
	      /**/ if (kA == A0[2]) AddFixa(element,b[2][0],a);
	      else if (kA == A1[2]) AddFixa(element,b[2][1],a);
	      a++;
	    }
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
    PetscInt f,n;
    n = element->nfix;
    for (f=0; f<n; f++) {
      PetscInt k = element->ifix[f];
      element->ufix[f] = U[k];
      U[k] = element->vfix[f];
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
  if (!element->collocation) {
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
  } else {
    PetscInt M = element->neq * element->dof;
    PetscInt N = element->nen * element->dof;
    PetscInt f,n;
    n = element->nfix;
    for (f=0; f<n; f++) {
      PetscInt k = element->ifix[f];
      PetscInt a = k / element->dof;
      if (element->index == element->colmap[a]) {
	PetscInt    c = k % element->dof;
	PetscScalar v = element->vfix[f];
	PetscInt i,j;
	for (i=0; i<M; i++) K[i*N+k] = 0.0;
	for (j=0; j<N; j++) K[c*N+j] = 0.0;
	K[c*N+k] = 1.0;
	F[c] = v;
      }
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
  if (!element->collocation) {
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
  } else {
    PetscInt f,n;
    n = element->nfix;
    for (f=0; f<n; f++) {
      PetscInt k = element->ifix[f];
      PetscInt a = k / element->dof;
      if (element->index == element->colmap[a]) {
	PetscInt    c = k % element->dof;
	PetscScalar v = element->vfix[f];
	PetscScalar u = element->ufix[f];
	F[c] = u - v;
      }
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
  if (!element->collocation) {
    PetscInt M = element->neq * element->dof;
    PetscInt N = element->nen * element->dof;
    PetscInt f,n;
    n = element->nfix;
    for (f=0; f<n; f++) {
      PetscInt k = element->ifix[f];
      PetscInt i,j;
      for (i=0; i<M; i++) J[i*N+k] = 0.0;
      for (j=0; j<N; j++) J[k*N+j] = 0.0;
      J[k*N+k] = 1.0;
    }
  } else {
    PetscInt M = element->neq * element->dof;
    PetscInt N = element->nen * element->dof;
    PetscInt f,n;
    n = element->nfix;
    for (f=0; f<n; f++) {
      PetscInt k = element->ifix[f];
      PetscInt a = k / element->dof;
      if (element->index == element->colmap[a]) {
	PetscInt c = k % element->dof;
	PetscInt i,j;
	if (element->index != element->colmap[a]) continue;
	for (i=0; i<M; i++) J[i*N+k] = 0.0;
	for (j=0; j<N; j++) J[c*N+j] = 0.0;
	J[c*N+k] = 1.0;
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAElementAssembleVec"
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

#undef  __FUNCT__
#define __FUNCT__ "IGAElementAssembleMat"
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
