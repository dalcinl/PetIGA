#include "petiga.h"
#include "petigagrid.h"

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
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
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
      PetscInt  p,m;
      PetscReal *U;
      ierr = PetscViewerBinaryRead(viewer,buf,3,PETSC_INT);CHKERRQ(ierr);
      p = buf[1];
      m = buf[2]-1;
      ierr = IGAGetAxis(iga,i,&axis);CHKERRQ(ierr);
      ierr = IGAAxisSetDegree(axis,p);CHKERRQ(ierr);CHKERRQ(ierr);
      ierr = IGAAxisSetKnots(axis,m,0);CHKERRQ(ierr);CHKERRQ(ierr);
      ierr = IGAAxisGetKnots(axis,0,&U);CHKERRQ(ierr);CHKERRQ(ierr);
      ierr = PetscViewerBinaryRead(viewer,U,m+1,PETSC_REAL);CHKERRQ(ierr);
    }
    ierr = IGASetUp(iga);CHKERRQ(ierr);
    if (kind) {
      ierr = IGALoadGeometry(iga,viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGALoadGeometry"
PetscErrorCode IGALoadGeometry(IGA iga,PetscViewer viewer)
{
  PetscBool      isbinary;
  PetscBool      skipheader;
  PetscInt       dim;
  PetscReal      min_w,max_w;
  Vec            nvec,gvec,lvec;
  VecScatter     g2n,g2l;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(iga,1,viewer,2);
  IGACheckSetUp(iga,1);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (!isbinary) SETERRQ(((PetscObject)viewer)->comm,PETSC_ERR_ARG_WRONG,"Only for binary viewers");
  ierr = PetscViewerBinaryGetSkipHeader(viewer,&skipheader);CHKERRQ(ierr);

  ierr = IGAGetSpatialDim(iga,&dim);CHKERRQ(ierr);
  {
    MPI_Comm comm = ((PetscObject)iga)->comm;
    PetscInt bs      = dim+1;
    PetscInt *sizes  = iga->geom_sizes;
    PetscInt *lstart = iga->geom_lstart;
    PetscInt *lwidth = iga->geom_lwidth;
    PetscInt *gstart = iga->geom_gstart;
    PetscInt *gwidth = iga->geom_gwidth;
    IGA_Grid grid;
    ierr = IGA_Grid_Create(comm,&grid);CHKERRQ(ierr);
    ierr = IGA_Grid_Init(grid,iga->dim,bs,sizes,lstart,lwidth,gstart,gwidth);CHKERRQ(ierr);
    ierr = IGA_Grid_GetVecGlobal(grid,iga->vectype,&gvec);CHKERRQ(ierr);
    ierr = IGA_Grid_GetVecLocal (grid,iga->vectype,&lvec);CHKERRQ(ierr);
    ierr = IGA_Grid_GetScatterG2N(grid,&g2n);CHKERRQ(ierr);
    ierr = IGA_Grid_GetScatterG2L(grid,&g2l);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)gvec);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)lvec);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)g2n);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)g2l);CHKERRQ(ierr);
    ierr = IGA_Grid_Destroy(&grid);CHKERRQ(ierr);
  }
  ierr = PetscObjectReference((PetscObject)lvec);CHKERRQ(ierr);
  ierr = VecDestroy(&iga->vec_geom);CHKERRQ(ierr);
  iga->vec_geom = lvec;

  ierr = VecDuplicate(gvec,&nvec);;CHKERRQ(ierr);
  /* viewer -> natural*/
  if (!skipheader)
    {ierr = VecLoad(nvec,viewer);CHKERRQ(ierr);}
  else
    {ierr = VecLoad_Binary_SkipHeader(nvec,viewer);CHKERRQ(ierr);}
  /* natural -> global */
  ierr = VecScatterBegin(g2n,nvec,gvec,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd  (g2n,nvec,gvec,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  /* global -> local */
  ierr = VecScatterBegin(g2l,gvec,lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (g2l,gvec,lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecStrideMin(gvec,dim,PETSC_NULL,&min_w);CHKERRQ(ierr);
  ierr = VecStrideMax(gvec,dim,PETSC_NULL,&max_w);CHKERRQ(ierr);

  ierr = VecScatterDestroy(&g2n);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&g2l);CHKERRQ(ierr);
  ierr = VecDestroy(&lvec);CHKERRQ(ierr);
  ierr = VecDestroy(&gvec);CHKERRQ(ierr);
  ierr = VecDestroy(&nvec);CHKERRQ(ierr);

  iga->geometry = PETSC_TRUE;
  iga->rational = (PetscAbs(max_w-min_w) > 100*PETSC_MACHINE_EPSILON) ? PETSC_TRUE : PETSC_FALSE;
  {
    PetscInt n,bs;
    PetscInt nnp,dim;
    PetscInt a,i,pos;
    const PetscScalar *Xw;
    PetscReal *X,*W;
    ierr = VecGetSize(iga->vec_geom,&n);CHKERRQ(ierr);
    ierr = VecGetBlockSize(iga->vec_geom,&bs);CHKERRQ(ierr);
    ierr = VecGetArrayRead(iga->vec_geom,&Xw);CHKERRQ(ierr);
    nnp = n / bs; dim = bs - 1;
    ierr = PetscMalloc1(nnp*dim,PetscReal,&iga->geometryX);CHKERRQ(ierr);
    ierr = PetscMalloc1(nnp,    PetscReal,&iga->geometryW);CHKERRQ(ierr);
    X = iga->geometryX;
    W = iga->geometryW;
    for (pos=0,a=0; a<nnp; a++) {
      for (i=0; i<dim; i++)
        X[i+a*dim] = PetscRealPart(Xw[pos++]);
      W[a] = PetscRealPart(Xw[pos++]);
      if (W[a] != 0.0)
        for (i=0; i<dim; i++)
          X[i+a*dim] /= W[a];
    }
    ierr = VecRestoreArrayRead(iga->vec_geom,&Xw);CHKERRQ(ierr);
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
  IGACheckSetUp(iga,1);

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
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (!isbinary) SETERRQ(((PetscObject)viewer)->comm,PETSC_ERR_ARG_WRONG,"Only for binary viewers");
  ierr = PetscViewerBinaryGetSkipHeader(viewer,&skipheader);CHKERRQ(ierr);

  { /* */
    PetscInt i=0,buf[3];
    PetscInt kind,dim;
    kind = iga->geometry ? (1 + (PetscInt)iga->rational) : 0;
    ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
    if (!skipheader) buf[i++] = IGA_FILE_CLASSID;
    buf[i++] = kind; buf[i++] = dim;
    ierr = PetscViewerBinaryWrite(viewer,buf,i,PETSC_INT,PETSC_TRUE);CHKERRQ(ierr);
    for (i=0; i<dim; i++) {
      IGAAxis   axis;
      PetscInt  p,m;
      PetscReal *U;
      ierr = IGAGetAxis(iga,i,&axis);CHKERRQ(ierr);
      ierr = IGAAxisGetDegree(axis,&p);CHKERRQ(ierr);
      ierr = IGAAxisGetKnots(axis,&m,&U);CHKERRQ(ierr);
      buf[0] = 0; /* XXX Unused! */
      buf[1] = p;
      buf[2] = m+1;
      ierr = PetscViewerBinaryWrite(viewer,buf,3,PETSC_INT,PETSC_TRUE);CHKERRQ(ierr);
      ierr = PetscViewerBinaryWrite(viewer,U,m+1,PETSC_REAL,PETSC_FALSE);CHKERRQ(ierr);
    }
  }
  if (iga->geometry && iga->vec_geom) {
    PetscInt   dim;
    Vec        nvec,gvec,lvec;
    VecScatter l2g,g2n;

    ierr = IGAGetSpatialDim(iga,&dim);CHKERRQ(ierr);
    {
      MPI_Comm comm = ((PetscObject)iga)->comm;
      PetscInt bs      = dim+1;
      PetscInt *sizes  = iga->geom_sizes;
      PetscInt *lstart = iga->geom_lstart;
      PetscInt *lwidth = iga->geom_lwidth;
      PetscInt *gstart = iga->geom_gstart;
      PetscInt *gwidth = iga->geom_gwidth;
      IGA_Grid grid;
      ierr = IGA_Grid_Create(comm,&grid);CHKERRQ(ierr);
      ierr = IGA_Grid_Init(grid,iga->dim,bs,sizes,lstart,lwidth,gstart,gwidth);CHKERRQ(ierr);
      ierr = IGA_Grid_GetVecGlobal (grid,iga->vectype,&gvec);CHKERRQ(ierr);
      ierr = IGA_Grid_GetVecLocal (grid,iga->vectype,&lvec);CHKERRQ(ierr);
      ierr = IGA_Grid_GetScatterL2G(grid,&l2g);CHKERRQ(ierr);
      ierr = IGA_Grid_GetScatterG2N(grid,&g2n);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)gvec);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)lvec);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)l2g);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)g2n);CHKERRQ(ierr);
      ierr = IGA_Grid_Destroy(&grid);CHKERRQ(ierr);
    }
    ierr = VecDuplicate(gvec,&nvec);;CHKERRQ(ierr);
    ierr = VecCopy(iga->vec_geom,lvec);CHKERRQ(ierr);

    /* local -> global */
    ierr = VecScatterBegin(l2g,lvec,gvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd  (l2g,lvec,gvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    /* global -> naural */
    ierr = VecScatterBegin(g2n,gvec,nvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd  (g2n,gvec,nvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    /* natural -> viewer */
    ierr = VecView(nvec,viewer);CHKERRQ(ierr);

    ierr = VecScatterDestroy(&g2n);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&l2g);CHKERRQ(ierr);
    ierr = VecDestroy(&lvec);CHKERRQ(ierr);
    ierr = VecDestroy(&gvec);CHKERRQ(ierr);
    ierr = VecDestroy(&nvec);CHKERRQ(ierr);
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGARead"
/*@
   IGARead - reads a IGA which has been saved in binary format
   
   Collective on IGA

   Input Parameters:
+  iga - the IGA context
-  filename - the file name which contains the IGA information

   Level: normal

.keywords: IGA, read
@*/
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
/*@
   IGAWrite - writes a IGA to a file in binary format
   
   Collective on IGA

   Input Parameters:
+  iga - the IGA context
-  filename - the file name in which the IGA information is saved

   Level: normal

.keywords: IGA, write
@*/
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
      const PetscInt *range = vec->map->range; /* XXX */
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
