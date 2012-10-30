#include "petiga.h"

#undef  __FUNCT__
#define __FUNCT__ "IGABoundaryCreate"
PetscErrorCode IGABoundaryCreate(IGABoundary *boundary)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(boundary,1);
  ierr = PetscNew(struct _n_IGABoundary,boundary);CHKERRQ(ierr);
  (*boundary)->refct = 1;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGABoundaryDestroy"
PetscErrorCode IGABoundaryDestroy(IGABoundary *_boundary)
{
  IGABoundary    boundary;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(_boundary,1);
  boundary = *_boundary; *_boundary = 0;
  if (!boundary) PetscFunctionReturn(0);
  if (--boundary->refct > 0) PetscFunctionReturn(0);
  ierr = IGABoundaryReset(boundary);CHKERRQ(ierr);
  ierr = PetscFree(boundary);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGABoundaryReset"
PetscErrorCode IGABoundaryReset(IGABoundary boundary)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (!boundary) PetscFunctionReturn(0);
  PetscValidPointer(boundary,1);
  boundary->dof = 0;
  boundary->count = 0;
  ierr = PetscFree(boundary->field);CHKERRQ(ierr);
  ierr = PetscFree(boundary->value);CHKERRQ(ierr);
  boundary->nload = 0;
  ierr = PetscFree(boundary->iload);CHKERRQ(ierr);
  ierr = PetscFree(boundary->vload);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGABoundaryReference"
PetscErrorCode IGABoundaryReference(IGABoundary boundary)
{
  PetscFunctionBegin;
  PetscValidPointer(boundary,1);
  boundary->refct++;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGABoundaryInit"
PetscErrorCode IGABoundaryInit(IGABoundary boundary,PetscInt dof)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidPointer(boundary,1);
  ierr = IGABoundaryReset(boundary);CHKERRQ(ierr);
  boundary->dof = dof;
  boundary->count = 0;
  ierr = PetscMalloc1(dof,PetscInt,   &boundary->field);CHKERRQ(ierr);
  ierr = PetscMalloc1(dof,PetscScalar,&boundary->value);CHKERRQ(ierr);
  boundary->nload = 0;
  ierr = PetscMalloc1(dof,PetscInt,   &boundary->iload);CHKERRQ(ierr);
  ierr = PetscMalloc1(dof,PetscScalar,&boundary->vload);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGABoundaryClear"
PetscErrorCode IGABoundaryClear(IGABoundary boundary)
{
  PetscFunctionBegin;
  PetscValidPointer(boundary,1);
  boundary->count   = 0;
  boundary->nload = 0;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGABoundarySetValue"
/*@
   IGABoundarySetValue - Used to set a constant Dirichlet boundary
   condition on the given boundary.
   
   Logically Collective on IGABoundary

   Input Parameters:
+  boundary - the IGAAxis context
.  field - the index of the field on which to enforce the condition
-  value - the value to set

   Level: normal

.keywords: IGA, boundary, Dirichlet
@*/
PetscErrorCode IGABoundarySetValue(IGABoundary boundary,PetscInt field,PetscScalar value)
{
  PetscInt dof;
  PetscFunctionBegin;
  PetscValidPointer(boundary,1);
  dof = boundary->dof;
  if (field <  0)   SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Field %D must be nonnegative",field);
  if (field >= dof) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Field %D, but dof %D",field,dof);
  { /**/
    PetscInt pos;
    for (pos=0; pos<boundary->count; pos++)
      if (boundary->field[pos] == field) break;
    if (pos==boundary->count) boundary->count++;
    boundary->field[pos] = field;
    boundary->value[pos] = value;
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGABoundarySetLoad"
/*@
   IGABoundarySetLoad - Used to set a constant Neumann boundary
   condition on the given boundary.
   
   Logically Collective on IGABoundary

   Input Parameters:
+  boundary - the IGAAxis context
.  field - the index of the field on which to enforce the condition
-  value - the value to set

   Level: normal

.keywords: IGA, boundary, Neumann
@*/
PetscErrorCode IGABoundarySetLoad(IGABoundary boundary,PetscInt field,PetscScalar value)
{
  PetscInt dof;
  PetscFunctionBegin;
  PetscValidPointer(boundary,1);
  dof = boundary->dof;
  if (field <  0)   SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Field %D must be nonnegative",field);
  if (field >= dof) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Field %D, but dof %D",field,dof);
  { /**/
    PetscInt pos;
    for (pos=0; pos<boundary->nload; pos++)
      if (boundary->iload[pos] == field) break;
    if (pos==boundary->nload) boundary->nload++;
    boundary->iload[pos] = field;
    boundary->vload[pos] = value;
  }
  PetscFunctionReturn(0);
}
