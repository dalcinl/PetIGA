#include "petiga.h"

extern PetscLogEvent IGA_FormScalar;

#undef  __FUNCT__
#define __FUNCT__ "IGAFormScalar"
PetscErrorCode IGAFormScalar(IGA iga,Vec vecU,PetscInt n,PetscScalar S[],
                             IGAUserScalar Scalar,void *ctx)
{
  MPI_Comm          comm;
  Vec               localU;
  const PetscScalar *arrayU;
  PetscScalar       *localS,*workS;
  IGAElement        element;
  IGAPoint          point;
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vecU,VEC_CLASSID,2);
  PetscValidScalarPointer(S,3);
  IGACheckSetUp(iga,1);

  ierr = PetscMalloc2(n,PetscScalar,&localS,n,PetscScalar,&workS);CHKERRQ(ierr);
  ierr = PetscMemzero(localS,n*sizeof(PetscScalar));CHKERRQ(ierr);

  /* Get local vector U and array */
  ierr = IGAGetLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);

  /* Element loop */
  ierr = PetscLogEventBegin(IGA_FormScalar,iga,vecU,0,0);CHKERRQ(ierr);
  ierr = IGAGetElement(iga,&element);CHKERRQ(ierr);
  ierr = IGAElementBegin(element);CHKERRQ(ierr);
  while (IGAElementNext(element)) {
    PetscScalar *U;
    ierr = IGAElementGetWorkVec(element,&U);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayU,U);CHKERRQ(ierr);
    /* Quadrature loop */
    ierr = IGAElementGetPoint(element,&point);CHKERRQ(ierr);
    ierr = IGAPointBegin(point);CHKERRQ(ierr);
    while (IGAPointNext(point)) {
      ierr = PetscMemzero(workS,n*sizeof(PetscScalar));CHKERRQ(ierr);
      ierr = Scalar(point,U,n,workS,ctx);CHKERRQ(ierr);
      ierr = IGAPointAddArray(point,n,workS,localS);CHKERRQ(ierr);
    }
  }
  ierr = IGAElementEnd(element);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(IGA_FormScalar,iga,vecU,0,0);CHKERRQ(ierr);

  /* Restore local vector U and array */
  ierr = IGARestoreLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);

  /* Assemble global scalars S */
  ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(localS,S,n,MPIU_SCALAR,MPIU_SUM,comm);CHKERRQ(ierr);

  ierr = PetscFree2(localS,workS);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
