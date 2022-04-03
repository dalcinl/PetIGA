#include "petiga.h"

#ifndef PCH2OPUS
#define "h2opus"
#endif

PetscErrorCode IGAPreparePCH2OPUS(IGA iga,PC pc)
{
  PetscBool         ish2opus;
  Vec               coords;
  PetscInt          i,n,dim;
  const PetscScalar *array;
  PetscReal         *xyz;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(pc,PC_CLASSID,2);
  IGACheckSetUpStage2(iga,1);

  ierr = PetscObjectTypeCompare((PetscObject)pc,PCH2OPUS,&ish2opus);CHKERRQ(ierr);
  if (!ish2opus) PetscFunctionReturn(0);

  ierr = IGACreateCoordinates(iga,&coords);CHKERRQ(ierr);
  ierr = VecGetLocalSize(coords,&n);CHKERRQ(ierr);
  ierr = VecGetBlockSize(coords,&dim);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coords,&array);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&xyz);CHKERRQ(ierr);
  for (i=0; i<n; i++) xyz[i] = PetscRealPart(array[i]);
  ierr = VecRestoreArrayRead(coords,&array);CHKERRQ(ierr);
  ierr = PCSetCoordinates(pc,dim,n/dim,xyz);CHKERRQ(ierr);
  ierr = PetscFree(xyz);CHKERRQ(ierr);
  ierr = VecDestroy(&coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
