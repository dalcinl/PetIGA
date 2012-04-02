#include "petiga.h"

static PetscErrorCode VecLoad_Binary_SkipHeader(Vec,PetscViewer);

#undef  __FUNCT__
#define __FUNCT__ "IGALoad"
PetscErrorCode IGALoad(IGA iga,PetscViewer viewer)
{
  PetscBool      isbinary;
  PetscBool      skipheader;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(iga,1,viewer,2);

  /* */
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (!isbinary) SETERRQ(((PetscObject)viewer)->comm,PETSC_ERR_ARG_WRONG,"Only for binary viewers");
  ierr = PetscViewerBinaryGetSkipHeader(viewer,&skipheader);CHKERRQ(ierr);

  ierr = IGAReset(iga);CHKERRQ(ierr);
  { /* */
    PetscInt i,buf[3];
    PetscInt kind,dim;
    if (!skipheader) {
      PetscInt classid;
      ierr = PetscViewerBinaryRead(viewer,&classid,1,PETSC_INT);CHKERRQ(ierr);
      if (classid != IGA_FILE_CLASSID) SETERRQ(((PetscObject)viewer)->comm,PETSC_ERR_ARG_WRONG,"Not an IGA in file");
    }
    ierr = PetscViewerBinaryRead(viewer,&kind,1,PETSC_INT);CHKERRQ(ierr);
    ierr = PetscViewerBinaryRead(viewer,&dim, 1,PETSC_INT);CHKERRQ(ierr);
    ierr = IGASetDim(iga,dim);CHKERRQ(ierr);
    for (i=0; i<dim; i++) {
      IGAAxis   axis;
      PetscBool periodic;
      PetscInt  p,m;
      PetscReal *U;
      ierr = PetscViewerBinaryRead(viewer,buf,3,PETSC_INT);CHKERRQ(ierr);
      periodic = buf[0] ? PETSC_TRUE : PETSC_FALSE;
      p = buf[1];
      m = buf[2]-1;
      ierr = IGAGetAxis(iga,i,&axis);CHKERRQ(ierr);
      ierr = IGAAxisSetPeriodic(axis,periodic);CHKERRQ(ierr);
      ierr = IGAAxisSetDegree(axis,p);CHKERRQ(ierr);CHKERRQ(ierr);
      ierr = IGAAxisSetKnots(axis,m,0);CHKERRQ(ierr);CHKERRQ(ierr);
      ierr = IGAAxisGetKnots(axis,0,&U);CHKERRQ(ierr);CHKERRQ(ierr);
      ierr = PetscViewerBinaryRead(viewer,U,m+1,PETSC_REAL);CHKERRQ(ierr);
    }
    if (kind) {
      DM  dm_geom;
      Vec vec_geom_global,vec_geom_local;
      PetscScalar min_w,max_w;
      ierr = IGACreateGeomDM(iga,&dm_geom);CHKERRQ(ierr);
      ierr = DMCreateGlobalVector(dm_geom,&vec_geom_global);CHKERRQ(ierr);
      if (!skipheader) {
        ierr = VecLoad(vec_geom_global,viewer);CHKERRQ(ierr);
      } else {
        Vec vec_geom_natural;
        ierr = DMDACreateNaturalVector(dm_geom,&vec_geom_natural);CHKERRQ(ierr);
        ierr = VecLoad_Binary_SkipHeader(vec_geom_natural,viewer);CHKERRQ(ierr);
        ierr = DMDANaturalToGlobalBegin(dm_geom,vec_geom_natural,INSERT_VALUES,vec_geom_global);CHKERRQ(ierr);
        ierr = DMDANaturalToGlobalEnd  (dm_geom,vec_geom_natural,INSERT_VALUES,vec_geom_global);CHKERRQ(ierr);
        ierr = VecDestroy(&vec_geom_natural);CHKERRQ(ierr);
      }
      ierr = DMCreateLocalVector(dm_geom,&vec_geom_local);CHKERRQ(ierr);
      ierr = DMGlobalToLocalBegin(dm_geom,vec_geom_global,INSERT_VALUES,vec_geom_local);CHKERRQ(ierr);
      ierr = DMGlobalToLocalEnd  (dm_geom,vec_geom_global,INSERT_VALUES,vec_geom_local);CHKERRQ(ierr);

      ierr = VecDestroy(&iga->vec_geom);CHKERRQ(ierr);
      ierr = VecDuplicate(vec_geom_local,&iga->vec_geom);CHKERRQ(ierr);
      ierr = VecCopy(vec_geom_local,iga->vec_geom);CHKERRQ(ierr);
      ierr = VecStrideMin(vec_geom_global,dim,PETSC_NULL,&min_w);CHKERRQ(ierr);
      ierr = VecStrideMax(vec_geom_global,dim,PETSC_NULL,&max_w);CHKERRQ(ierr);
      iga->rational = (PetscAbs(max_w-min_w) > 100*PETSC_MACHINE_EPSILON) ? PETSC_TRUE : PETSC_FALSE;

      ierr = VecDestroy(&vec_geom_global);CHKERRQ(ierr);
      ierr = VecDestroy(&vec_geom_local);CHKERRQ(ierr);
      ierr = DMDestroy(&dm_geom);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASave"
PetscErrorCode IGASave(IGA iga,PetscViewer viewer)
{
  PetscBool      isbinary;
  PetscBool      skipheader;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);

  /* */
  if (viewer) {
    PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
    PetscCheckSameComm(iga,1,viewer,2);
  } else {
    MPI_Comm comm;
    ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
    viewer = PETSC_VIEWER_BINARY_(comm);
    if (!viewer) PetscFunctionReturn(PETSC_ERR_PLIB);
  }

  /* */
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (!isbinary) SETERRQ(((PetscObject)viewer)->comm,PETSC_ERR_ARG_WRONG,"Only for binary viewers");
  ierr = PetscViewerBinaryGetSkipHeader(viewer,&skipheader);CHKERRQ(ierr);

  { /* */
    PetscInt i=0,buf[3];
    PetscInt kind,dim;
    kind = iga->vec_geom ? (1 + (PetscInt)iga->rational) : 0;
    ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
    if (!skipheader) buf[i++] = IGA_FILE_CLASSID;
    buf[i++] = kind; buf[i++] = dim;
    ierr = PetscViewerBinaryWrite(viewer,buf,i,PETSC_INT,PETSC_TRUE);CHKERRQ(ierr);
    for (i=0; i<dim; i++) {
      IGAAxis   axis;
      PetscBool periodic;
      PetscInt  p,m;
      PetscReal *U;
      ierr = IGAGetAxis(iga,i,&axis);CHKERRQ(ierr);
      ierr = IGAAxisGetPeriodic(axis,&periodic);CHKERRQ(ierr);
      ierr = IGAAxisGetDegree(axis,&p);CHKERRQ(ierr);
      ierr = IGAAxisGetKnots(axis,&m,&U);CHKERRQ(ierr);
      buf[0] = periodic;
      buf[1] = p;
      buf[2] = m+1;
      ierr = PetscViewerBinaryWrite(viewer,buf,3,PETSC_INT,PETSC_TRUE);CHKERRQ(ierr);
      ierr = PetscViewerBinaryWrite(viewer,U,m+1,PETSC_REAL,PETSC_FALSE);CHKERRQ(ierr);
    }
    if (iga->vec_geom) {
      DM  dm_geom;
      Vec vec_geom_global,vec_geom_local;
      ierr = IGACreateGeomDM(iga,&dm_geom);CHKERRQ(ierr);
      ierr = DMCreateLocalVector(dm_geom,&vec_geom_local);CHKERRQ(ierr);
      ierr = VecCopy(iga->vec_geom,vec_geom_local);CHKERRQ(ierr);
      ierr = DMCreateGlobalVector(dm_geom,&vec_geom_global);CHKERRQ(ierr);
      ierr = DMLocalToGlobalBegin(dm_geom,vec_geom_local,INSERT_VALUES,vec_geom_global);CHKERRQ(ierr);
      ierr = DMLocalToGlobalEnd  (dm_geom,vec_geom_local,INSERT_VALUES,vec_geom_global);CHKERRQ(ierr);
      ierr = VecView(vec_geom_global,viewer);CHKERRQ(ierr);
      ierr = VecDestroy(&vec_geom_global);CHKERRQ(ierr);
      ierr = VecDestroy(&vec_geom_local);CHKERRQ(ierr);
      ierr = DMDestroy(&dm_geom);CHKERRQ(ierr);
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
  PetscValidCharPointer(filename,2);
  ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
  ierr = PetscViewerCreate(comm,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer,PETSCVIEWERBINARY);CHKERRQ(ierr);
  ierr = PetscViewerBinarySkipInfo(viewer);CHKERRQ(ierr);
  /*ierr = PetscViewerBinarySetSkipHeader(viewer,PETSC_TRUE);CHKERRQ(ierr);*/
  ierr = PetscViewerFileSetMode(viewer,FILE_MODE_READ);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(viewer,filename);CHKERRQ(ierr);
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
  PetscValidCharPointer(filename,2);
  ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
  ierr = PetscViewerCreate(comm,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer,PETSCVIEWERBINARY);CHKERRQ(ierr);
  ierr = PetscViewerBinarySkipInfo(viewer);CHKERRQ(ierr);
  /*ierr = PetscViewerBinarySetSkipHeader(viewer,PETSC_TRUE);CHKERRQ(ierr);*/
  ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(viewer,filename);CHKERRQ(ierr);
  ierr = IGASave(iga,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "VecLoad_Binary_SkipHeader"
static PetscErrorCode VecLoad_Binary_SkipHeader(Vec vec, PetscViewer viewer)
{
  MPI_Comm       comm;
  PetscMPIInt    i,rank,size,tag;
  int            fd;
  PetscInt       n;
  PetscScalar    *array,*work;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = PetscCommGetNewTag(comm,&tag);CHKERRQ(ierr);
  ierr = PetscViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);

  ierr = VecGetLocalSize(vec,&n);CHKERRQ(ierr);
  ierr = VecGetArray(vec,&array);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscBinaryRead(fd,array,n,PETSC_SCALAR);CHKERRQ(ierr);
    if (size > 1) {
      const PetscInt *range = vec->map->range;
      n = 1;
      for (i=1; i<size; i++)
        n = PetscMax(n,range[i+1] - range[i]);
      ierr = PetscMalloc(n*sizeof(PetscScalar),&work);CHKERRQ(ierr);
      for (i=1; i<size; i++) {
        n   = range[i+1] - range[i];
        ierr = PetscBinaryRead(fd,work,n,PETSC_SCALAR);CHKERRQ(ierr);
        ierr = MPI_Send(work,n,MPIU_SCALAR,i,tag,comm);CHKERRQ(ierr);
      }
      ierr = PetscFree(work);CHKERRQ(ierr);
    }
  } else {
    MPI_Status status;
    ierr = MPI_Recv(array,n,MPIU_SCALAR,0,tag,comm,&status);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(vec,&array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
