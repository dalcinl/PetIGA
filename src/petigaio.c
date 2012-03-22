#include "petiga.h"

#define IGA_FILE_CLASSID DM_FILE_CLASSID

#undef  __FUNCT__
#define __FUNCT__ "IGALoad"
PetscErrorCode IGALoad(IGA iga,PetscViewer viewer)
{
  PetscBool      isbinary;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(iga,1,viewer,2);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (!isbinary) SETERRQ(((PetscObject)viewer)->comm,PETSC_ERR_ARG_WRONG,"Only for binary viewers");
  { /* */
    PetscInt i,buf[3];
    PetscInt classid,dim;
    ierr = PetscViewerBinaryRead(viewer,buf,2,PETSC_INT);CHKERRQ(ierr);
    classid = buf[0]; dim = buf[1];
    if (classid != IGA_FILE_CLASSID) SETERRQ(((PetscObject)viewer)->comm,PETSC_ERR_ARG_WRONG,"Not an IGA in file");
    ierr = IGASetDim(iga,dim);CHKERRQ(ierr);
    for (i=0; i<dim; i++) {
      IGAAxis   axis;
      PetscBool periodic;
      PetscInt  p,n,m;
      PetscReal *U;
      ierr = PetscViewerBinaryRead(viewer,&buf,3,PETSC_INT);CHKERRQ(ierr);
      periodic = buf[0] ? PETSC_TRUE : PETSC_FALSE;
      p = buf[1]; n = buf[2]; m = n+p+1;
      ierr = IGAGetAxis(iga,i,&axis);CHKERRQ(ierr);
      ierr = IGAAxisSetPeriodic(axis,periodic);CHKERRQ(ierr);
      ierr = IGAAxisSetOrder(axis,p);CHKERRQ(ierr);CHKERRQ(ierr);
      ierr = IGAAxisSetKnots(axis,m,0);CHKERRQ(ierr);CHKERRQ(ierr);
      ierr = IGAAxisGetKnots(axis,0,&U);CHKERRQ(ierr);CHKERRQ(ierr);
      ierr = PetscViewerBinaryRead(viewer,&U,m,PETSC_REAL);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASave"
PetscErrorCode IGASave(IGA iga,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);

  /* */
  if (viewer) {
    PetscBool isbinary;
    PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
    PetscCheckSameComm(iga,1,viewer,2);
    ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
    if (!isbinary) SETERRQ(((PetscObject)viewer)->comm,PETSC_ERR_ARG_WRONG,"Only for binary viewers");
  } else {
    MPI_Comm comm;
    ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
    viewer = PETSC_VIEWER_BINARY_(comm);
    if (!viewer) PetscFunctionReturn(PETSC_ERR_PLIB);
  }

  { /* */
    PetscInt i,buf[3];
    PetscInt dim;
    ierr = IGAGetDim(iga,&dim);
    buf[0] = IGA_FILE_CLASSID; buf[1] = dim;
    ierr = PetscViewerBinaryWrite(viewer,buf,2,PETSC_INT,PETSC_TRUE);CHKERRQ(ierr);
    for (i=0; i<dim; i++) {
      IGAAxis   axis;
      PetscBool periodic;
      PetscInt  p,n,m;
      PetscReal *U;
      ierr = IGAGetAxis(iga,i,&axis);CHKERRQ(ierr);
      ierr = IGAAxisGetPeriodic(axis,&periodic);CHKERRQ(ierr);
      ierr = IGAAxisGetOrder(axis,&p);CHKERRQ(ierr);
      ierr = IGAAxisGetKnots(axis,&m,&U);CHKERRQ(ierr);
      buf[0] = periodic; buf[1] = p; buf[2] = n = m-p-1;
      ierr = PetscViewerBinaryWrite(viewer,buf,3,PETSC_INT,PETSC_TRUE);CHKERRQ(ierr);
      ierr = PetscViewerBinaryWrite(viewer,U,m,PETSC_REAL,PETSC_FALSE);CHKERRQ(ierr);
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGARead"
PetscErrorCode IGARead(IGA iga,const char filename[])
{
  MPI_Comm       comm;
  PetscViewer    viewer;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(comm,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = IGALoad(iga,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAWrite"
PetscErrorCode IGAWrite(IGA iga,const char filename[])
{
  MPI_Comm       comm;
  PetscViewer    viewer;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(comm,filename,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = IGASave(iga,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
