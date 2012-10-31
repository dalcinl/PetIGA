#include "petiga.h"
#include "petigagrid.h"

extern PetscErrorCode IGASetUp_Basic(IGA);
static PetscErrorCode VecLoad_Binary_SkipHeader(Vec,PetscViewer);

#undef  __FUNCT__
#define __FUNCT__ "IGALoad"
PetscErrorCode IGALoad(IGA iga,PetscViewer viewer)
{
  PetscBool      isbinary;
  PetscBool      skipheader;
  PetscInt       descr;
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

  if (!skipheader) {
    PetscInt classid = 0;
    ierr = PetscViewerBinaryRead(viewer,&classid,1,PETSC_INT);CHKERRQ(ierr);
    if (classid != IGA_FILE_CLASSID) SETERRQ(((PetscObject)viewer)->comm,PETSC_ERR_ARG_WRONG,"Not an IGA in file");
  }

  ierr = PetscViewerBinaryRead(viewer,&descr,1,PETSC_INT);CHKERRQ(ierr);
  if (descr >= 0) { /* */
    PetscInt i,dim;
    ierr = PetscViewerBinaryRead(viewer,&dim, 1,PETSC_INT);CHKERRQ(ierr);
    ierr = IGASetDim(iga,dim);CHKERRQ(ierr);
    for (i=0; i<dim; i++) {
      IGAAxis   axis;
      PetscInt  p,m;
      PetscReal *U;
      ierr = PetscViewerBinaryRead(viewer,&p,1,PETSC_INT);CHKERRQ(ierr);
      ierr = PetscViewerBinaryRead(viewer,&m,1,PETSC_INT);CHKERRQ(ierr);
      ierr = IGAGetAxis(iga,i,&axis);CHKERRQ(ierr);
      ierr = IGAAxisSetDegree(axis,p);CHKERRQ(ierr);CHKERRQ(ierr);
      ierr = IGAAxisSetKnots(axis,m-1,PETSC_NULL);CHKERRQ(ierr);CHKERRQ(ierr);
      ierr = IGAAxisGetKnots(axis,PETSC_NULL,&U);CHKERRQ(ierr);CHKERRQ(ierr);
      ierr = PetscViewerBinaryRead(viewer,U,m,PETSC_REAL);CHKERRQ(ierr);
    }
    ierr = IGASetUp_Basic(iga);CHKERRQ(ierr);
  }
  if (PetscAbs(descr) >= 1) { /* */
    PetscInt nsd;
    ierr = PetscViewerBinaryRead(viewer,&nsd,1,PETSC_INT);CHKERRQ(ierr);
    ierr = IGASetSpatialDim(iga,nsd);CHKERRQ(ierr);
    ierr = IGALoadGeometry(iga,viewer);CHKERRQ(ierr);
  }
  ierr = IGASetUp(iga);CHKERRQ(ierr);
#if 0
  /* XXX waiting implementation ... */
  if (PetscAbs(descr) >= 2) { /* */
    PetscInt npd,ncd;
    ierr = PetscViewerBinaryRead(viewer,&npd,1,PETSC_INT);CHKERRQ(ierr);
    ierr = PetscViewerBinaryRead(viewer,&ncd,1,PETSC_INT);CHKERRQ(ierr);
    ierr = IGASetDataDim(iga,npd,ncd);CHKERRQ(ierr);
    if (npd > 0) {ierr = IGALoadPointData(iga,viewer);CHKERRQ(ierr);}
    if (ncd > 0) {ierr = IGALoadCellData(iga,viewer);CHKERRQ(ierr);}
  }
#endif
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

  if (viewer) {
    PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
    PetscCheckSameComm(iga,1,viewer,2);
  } else {
    MPI_Comm comm;
    ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
    viewer = PETSC_VIEWER_BINARY_(comm);
    if (!viewer) PetscFunctionReturn(PETSC_ERR_PLIB);
  }
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (!isbinary) SETERRQ(((PetscObject)viewer)->comm,PETSC_ERR_ARG_WRONG,"Only for binary viewers");
  ierr = PetscViewerBinaryGetSkipHeader(viewer,&skipheader);CHKERRQ(ierr);

  if (!skipheader) {
    PetscInt classid = IGA_FILE_CLASSID;
    ierr = PetscViewerBinaryWrite(viewer,&classid,1,PETSC_INT,PETSC_TRUE);CHKERRQ(ierr);
  }
  { /* */
    PetscInt descr = iga->geometry ? 1 : 0;
    ierr = PetscViewerBinaryWrite(viewer,&descr,1,PETSC_INT,PETSC_TRUE);CHKERRQ(ierr);
  }
  { /* */
    PetscInt i,dim;
    ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
    ierr = PetscViewerBinaryWrite(viewer,&dim,1,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
    for (i=0; i<dim; i++) {
      IGAAxis   axis;
      PetscInt  p,m,buf[2];
      PetscReal *U;
      ierr = IGAGetAxis(iga,i,&axis);CHKERRQ(ierr);
      ierr = IGAAxisGetDegree(axis,&p);CHKERRQ(ierr);
      ierr = IGAAxisGetKnots(axis,&m,&U);CHKERRQ(ierr);
      buf[0] = p; buf[1] = m+1;
      ierr = PetscViewerBinaryWrite(viewer,buf,2,PETSC_INT,PETSC_TRUE);CHKERRQ(ierr);
      ierr = PetscViewerBinaryWrite(viewer,U,m+1,PETSC_REAL,PETSC_FALSE);CHKERRQ(ierr);
    }
  }
  if (iga->geometry) {
    PetscInt nsd;
    ierr = IGAGetSpatialDim(iga,&nsd);CHKERRQ(ierr);
    ierr = PetscViewerBinaryWrite(viewer,&nsd,1,PETSC_INT,PETSC_TRUE);CHKERRQ(ierr);
    ierr = IGASaveGeometry(iga,viewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGA_NewGridGeom"
static PetscErrorCode IGA_NewGridGeom(IGA iga,PetscInt bs,IGA_Grid *grid)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(grid,3);
  {
    MPI_Comm comm    = ((PetscObject)iga)->comm;
    PetscInt dim     = iga->dim;
    PetscInt *sizes  = iga->geom_sizes;
    PetscInt *lstart = iga->geom_lstart;
    PetscInt *lwidth = iga->geom_lwidth;
    PetscInt *gstart = iga->geom_gstart;
    PetscInt *gwidth = iga->geom_gwidth;
    ierr = IGA_Grid_Create(comm,grid);CHKERRQ(ierr);
    ierr = IGA_Grid_Init(*grid,dim,bs,sizes,lstart,lwidth,gstart,gwidth);CHKERRQ(ierr);
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
  if (iga->setupstage < 1) IGACheckSetUp(iga,1);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (!isbinary) SETERRQ(((PetscObject)viewer)->comm,PETSC_ERR_ARG_WRONG,"Only for binary viewers");
  ierr = PetscViewerBinaryGetSkipHeader(viewer,&skipheader);CHKERRQ(ierr);

  ierr = IGAGetSpatialDim(iga,&dim);CHKERRQ(ierr);
  if (dim < 1)
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,
            "Must call IGASetSpatialDim() first");

  iga->geometry = PETSC_FALSE;
  iga->rational = PETSC_FALSE;
  ierr = PetscFree(iga->geometryX);CHKERRQ(ierr);
  ierr = PetscFree(iga->geometryW);CHKERRQ(ierr);
  ierr = VecDestroy(&iga->geom_vec);CHKERRQ(ierr);

  {
    IGA_Grid grid;
    ierr = IGA_NewGridGeom(iga,dim+1,&grid);CHKERRQ(ierr);
    ierr = IGA_Grid_GetVecNatural(grid,iga->vectype,&nvec);CHKERRQ(ierr);
    ierr = IGA_Grid_GetVecGlobal (grid,iga->vectype,&gvec);CHKERRQ(ierr);
    ierr = IGA_Grid_GetVecLocal  (grid,iga->vectype,&lvec);CHKERRQ(ierr);
    ierr = IGA_Grid_GetScatterG2N(grid,&g2n);CHKERRQ(ierr);
    ierr = IGA_Grid_GetScatterG2L(grid,&g2l);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)nvec);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)gvec);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)lvec);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)g2n);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)g2l);CHKERRQ(ierr);
    ierr = IGA_Grid_Destroy(&grid);CHKERRQ(ierr);
  }
  ierr = PetscObjectReference((PetscObject)lvec);CHKERRQ(ierr);
  iga->geom_vec = lvec;

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
    ierr = VecGetSize(iga->geom_vec,&n);CHKERRQ(ierr);
    ierr = VecGetBlockSize(iga->geom_vec,&bs);CHKERRQ(ierr);
    nnp = n / bs; dim = bs - 1;
    ierr = PetscMalloc1(nnp*dim,PetscReal,&iga->geometryX);CHKERRQ(ierr);
    ierr = PetscMalloc1(nnp,    PetscReal,&iga->geometryW);CHKERRQ(ierr);
    X = iga->geometryX; W = iga->geometryW;
    ierr = VecGetArrayRead(iga->geom_vec,&Xw);CHKERRQ(ierr);
    for (pos=0,a=0; a<nnp; a++) {
      for (i=0; i<dim; i++)
        X[i+a*dim] = PetscRealPart(Xw[pos++]);
      W[a] = PetscRealPart(Xw[pos++]);
      if (W[a] != 0.0)
        for (i=0; i<dim; i++)
          X[i+a*dim] /= W[a];
    }
    ierr = VecRestoreArrayRead(iga->geom_vec,&Xw);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASaveGeometry"
PetscErrorCode IGASaveGeometry(IGA iga,PetscViewer viewer)
{
  PetscBool      isbinary;
  PetscBool      skipheader;
  PetscInt       dim;
  Vec            nvec,gvec,lvec;
  VecScatter     l2g,g2n;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(iga,1,viewer,2);
  if (iga->setupstage < 1) IGACheckSetUp(iga,1);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (!isbinary) SETERRQ(((PetscObject)viewer)->comm,PETSC_ERR_ARG_WRONG,"Only for binary viewers");
  ierr = PetscViewerBinaryGetSkipHeader(viewer,&skipheader);CHKERRQ(ierr);

  ierr = IGAGetSpatialDim(iga,&dim);CHKERRQ(ierr);
  if (dim < 1)
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,
            "Must call IGASetSpatialDim() first");
  {
    IGA_Grid grid;
    ierr = IGA_NewGridGeom(iga,dim+1,&grid);CHKERRQ(ierr);
    ierr = IGA_Grid_GetVecNatural(grid,iga->vectype,&nvec);CHKERRQ(ierr);
    ierr = IGA_Grid_GetVecGlobal (grid,iga->vectype,&gvec);CHKERRQ(ierr);
    ierr = IGA_Grid_GetVecLocal  (grid,iga->vectype,&lvec);CHKERRQ(ierr);
    ierr = IGA_Grid_GetScatterL2G(grid,&l2g);CHKERRQ(ierr);
    ierr = IGA_Grid_GetScatterG2N(grid,&g2n);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)nvec);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)gvec);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)lvec);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)l2g);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)g2n);CHKERRQ(ierr);
    ierr = IGA_Grid_Destroy(&grid);CHKERRQ(ierr);
  }
  ierr = VecCopy(iga->geom_vec,lvec);CHKERRQ(ierr);

  /* local -> global */
  ierr = VecScatterBegin(l2g,lvec,gvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (l2g,lvec,gvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  /* global -> natural */
  ierr = VecScatterBegin(g2n,gvec,nvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (g2n,gvec,nvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  /* natural -> viewer */
  ierr = VecView(nvec,viewer);CHKERRQ(ierr);

  ierr = VecScatterDestroy(&g2n);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&l2g);CHKERRQ(ierr);
  ierr = VecDestroy(&lvec);CHKERRQ(ierr);
  ierr = VecDestroy(&gvec);CHKERRQ(ierr);
  ierr = VecDestroy(&nvec);CHKERRQ(ierr);

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
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
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

#undef  __FUNCT__
#define __FUNCT__ "IGALoadVec"
PetscErrorCode IGALoadVec(IGA iga,Vec vec,PetscViewer viewer)
{
  Vec            natural;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,3);
  PetscCheckSameComm(iga,1,vec,2);
  PetscCheckSameComm(iga,1,viewer,3);
  IGACheckSetUp(iga,1);

  ierr = IGAGetNaturalVec(iga,&natural);
  ierr = VecLoad(natural,viewer);CHKERRQ(ierr);
  ierr = IGANaturalToGlobal(iga,natural,vec);

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGASaveVec"
PetscErrorCode IGASaveVec(IGA iga,Vec vec,PetscViewer viewer)
{
  Vec            natural;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,3);
  PetscCheckSameComm(iga,1,vec,2);
  PetscCheckSameComm(iga,1,viewer,3);
  IGACheckSetUp(iga,1);

  ierr = IGAGetNaturalVec(iga,&natural);
  ierr = IGAGlobalToNatural(iga,vec,natural);
  ierr = PetscObjectSetName((PetscObject)natural,((PetscObject)vec)->name);CHKERRQ(ierr);
  ierr = VecView(natural,viewer);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAReadVec"
PetscErrorCode IGAReadVec(IGA iga,Vec vec,const char filename[])
{
  MPI_Comm       comm;
  PetscViewer    viewer;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,1);
  PetscCheckSameComm(iga,1,vec,2);
  PetscValidCharPointer(filename,2);
  IGACheckSetUp(iga,1);

  ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
  ierr = PetscViewerCreate(comm,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer,PETSCVIEWERBINARY);CHKERRQ(ierr);
  ierr = PetscViewerBinarySkipInfo(viewer);CHKERRQ(ierr);
  /*ierr = PetscViewerBinarySetSkipHeader(viewer,PETSC_TRUE);CHKERRQ(ierr);*/
  ierr = PetscViewerFileSetMode(viewer,FILE_MODE_READ);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(viewer,filename);CHKERRQ(ierr);
  ierr = IGALoadVec(iga,vec,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAWriteVec"
PetscErrorCode IGAWriteVec(IGA iga,Vec vec,const char filename[])
{
  MPI_Comm       comm;
  PetscViewer    viewer;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,1);
  PetscCheckSameComm(iga,1,vec,2);
  PetscValidCharPointer(filename,2);
  IGACheckSetUp(iga,1);

  ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
  ierr = PetscViewerCreate(comm,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer,PETSCVIEWERBINARY);CHKERRQ(ierr);
  ierr = PetscViewerBinarySkipInfo(viewer);CHKERRQ(ierr);
  /*ierr = PetscViewerBinarySetSkipHeader(viewer,PETSC_TRUE);CHKERRQ(ierr);*/
  ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(viewer,filename);CHKERRQ(ierr);
  ierr = IGASaveVec(iga,vec,viewer);CHKERRQ(ierr);
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
