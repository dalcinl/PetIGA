#include "petiga.h"
#include "petigagrid.h"

#if PETSC_VERSION_LT(3,13,0)
#define PetscViewerBinaryWrite(v,p,n,t) PetscViewerBinaryWrite(v,p,n,t,PETSC_FALSE)
#endif

PETSC_EXTERN PetscErrorCode IGASetUp_Basic(IGA);
static       PetscErrorCode VecLoad_Binary_SkipHeader(Vec,PetscViewer);

PetscErrorCode IGALoad(IGA iga,PetscViewer viewer)
{
  PetscBool      isbinary;
  PetscBool      skipheader;
  PetscBool      geometry;
  PetscBool      property;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(iga,1,viewer,2);

  /* */
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (!isbinary) SETERRQ(((PetscObject)viewer)->comm,PETSC_ERR_ARG_WRONG,"Only for binary viewers");
  ierr = PetscViewerBinaryGetSkipHeader(viewer,&skipheader);CHKERRQ(ierr);

  if (!skipheader) {
    PetscInt classid = 0;
    ierr = PetscViewerBinaryRead(viewer,&classid,1,NULL,PETSC_INT);CHKERRQ(ierr);
    if (classid != IGA_FILE_CLASSID) SETERRQ(((PetscObject)viewer)->comm,PETSC_ERR_ARG_WRONG,"Not an IGA in file");
  }
  { /* */
    PetscInt info = 0;
    ierr = PetscViewerBinaryRead(viewer,&info,1,NULL,PETSC_INT);CHKERRQ(ierr);
    geometry = (info & 0x1) ? PETSC_TRUE : PETSC_FALSE;
    property = (info & 0x2) ? PETSC_TRUE : PETSC_FALSE;
  }
  ierr = IGAReset(iga);CHKERRQ(ierr);
  { /* */
    PetscInt i,dim;
    ierr = PetscViewerBinaryRead(viewer,&dim,1,NULL,PETSC_INT);CHKERRQ(ierr);
    ierr = IGASetDim(iga,dim);CHKERRQ(ierr);
    for (i=0; i<dim; i++) {
      IGAAxis   axis;
      PetscInt  p,m;
      PetscReal *U;
      ierr = PetscViewerBinaryRead(viewer,&p,1,NULL,PETSC_INT);CHKERRQ(ierr);
      ierr = PetscViewerBinaryRead(viewer,&m,1,NULL,PETSC_INT);CHKERRQ(ierr);
      ierr = PetscMalloc1((size_t)m,&U);CHKERRQ(ierr);
      ierr = PetscViewerBinaryRead(viewer,U,m,NULL,PETSC_REAL);CHKERRQ(ierr);
      ierr = IGAGetAxis(iga,i,&axis);CHKERRQ(ierr);
      ierr = IGAAxisInit(axis,p,m-1,U);CHKERRQ(ierr);CHKERRQ(ierr);
      ierr = PetscFree(U);CHKERRQ(ierr);
    }
  }
  ierr = IGASetUp_Basic(iga);CHKERRQ(ierr);
  if (geometry) { /* */
    PetscInt dim;
    ierr = PetscViewerBinaryRead(viewer,&dim,1,NULL,PETSC_INT);CHKERRQ(ierr);
    ierr = IGASetGeometryDim(iga,dim);CHKERRQ(ierr);
    ierr = IGALoadGeometry(iga,viewer);CHKERRQ(ierr);
  }
  if (property) { /* */
    PetscInt dim;
    ierr = PetscViewerBinaryRead(viewer,&dim,1,NULL,PETSC_INT);CHKERRQ(ierr);
    ierr = IGASetPropertyDim(iga,dim);CHKERRQ(ierr);
    ierr = IGALoadProperty(iga,viewer);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode IGASave(IGA iga,PetscViewer viewer)
{
  PetscBool      isbinary;
  PetscBool      skipheader;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  IGACheckSetUpStage2(iga,1);

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
    ierr = PetscViewerBinaryWrite(viewer,&classid,1,PETSC_INT);CHKERRQ(ierr);
  }
  { /* */
    PetscInt info = 0;
    if (iga->geometry) info |= 0x1;
    if (iga->property) info |= 0x2;
    ierr = PetscViewerBinaryWrite(viewer,&info,1,PETSC_INT);CHKERRQ(ierr);
  }
  { /* */
    PetscInt i,dim;
    ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
    ierr = PetscViewerBinaryWrite(viewer,&dim,1,PETSC_INT);CHKERRQ(ierr);
    for (i=0; i<dim; i++) {
      IGAAxis   axis;
      PetscInt  p,m,buf[2];
      PetscReal *U;
      ierr = IGAGetAxis(iga,i,&axis);CHKERRQ(ierr);
      ierr = IGAAxisGetDegree(axis,&p);CHKERRQ(ierr);
      ierr = IGAAxisGetKnots(axis,&m,&U);CHKERRQ(ierr);
      buf[0] = p; buf[1] = m+1;
      ierr = PetscViewerBinaryWrite(viewer,buf,2,PETSC_INT);CHKERRQ(ierr);
      ierr = PetscViewerBinaryWrite(viewer,U,m+1,PETSC_REAL);CHKERRQ(ierr);
    }
  }
  if (iga->geometry) { /* */
    PetscInt dim;
    ierr = IGAGetGeometryDim(iga,&dim);CHKERRQ(ierr);
    ierr = PetscViewerBinaryWrite(viewer,&dim,1,PETSC_INT);CHKERRQ(ierr);
    ierr = IGASaveGeometry(iga,viewer);CHKERRQ(ierr);
  }
  if (iga->property) { /* */
    PetscInt dim;
    ierr = IGAGetPropertyDim(iga,&dim);CHKERRQ(ierr);
    ierr = PetscViewerBinaryWrite(viewer,&dim,1,PETSC_INT);CHKERRQ(ierr);
    ierr = IGASaveProperty(iga,viewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode IGA_NewGridIO(IGA iga,PetscInt bs,IGA_Grid *grid)
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

/*@
   IGASetGeometryDim - Sets the dimension of the geometry

   Logically Collective on IGA

   Input Parameters:
+  iga - the IGA context
-  dim - the dimension of the geometry

   Level: normal

.keywords: IGA, dimension
@*/
PetscErrorCode IGASetGeometryDim(IGA iga,PetscInt dim)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidLogicalCollectiveInt(iga,dim,2);
  if (dim < 1 || dim > 3)
    SETERRQ1(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,
             "Number of space dimensions must be in range [1,3], got %D",dim);
  if (dim < iga->dim)
    SETERRQ2(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,
             "Number of space dimensions must greater than or equal to %D, got %D",iga->dim,dim);
  if (iga->geometry == dim) PetscFunctionReturn(0);
  iga->geometry = dim;
  iga->rational = PETSC_FALSE;
  ierr = PetscFree(iga->geometryX);CHKERRQ(ierr);
  ierr = PetscFree(iga->rationalW);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGAGetGeometryDim(IGA iga,PetscInt *dim)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(dim,2);
  *dim = iga->geometry;
  PetscFunctionReturn(0);
}

PetscErrorCode IGALoadGeometry(IGA iga,PetscViewer viewer)
{
  PetscBool      isbinary;
  PetscBool      skipheader;
  PetscInt       nsd;
  PetscReal      min_w,max_w,tol_w = 100*PETSC_MACHINE_EPSILON;
  Vec            nvec,gvec,lvec;
  VecScatter     g2n,g2l;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(iga,1,viewer,2);
  IGACheckSetUpStage1(iga,1);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (!isbinary) SETERRQ(((PetscObject)viewer)->comm,PETSC_ERR_ARG_WRONG,"Only for binary viewers");
  ierr = PetscViewerBinaryGetSkipHeader(viewer,&skipheader);CHKERRQ(ierr);

  ierr = IGAGetGeometryDim(iga,&nsd);CHKERRQ(ierr);
  if (nsd < 1) SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_ARG_WRONGSTATE,
                       "Must call IGASetGeometryDim() first");
  {
    IGA_Grid grid;
    ierr = IGA_NewGridIO(iga,nsd+1,&grid);CHKERRQ(ierr);
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
  /* viewer -> natural*/
  if (!skipheader) {
    ierr = VecLoad(nvec,viewer);CHKERRQ(ierr);
  } else {
    ierr = VecLoad_Binary_SkipHeader(nvec,viewer);CHKERRQ(ierr);
  }
  /* natural -> global */
  ierr = VecScatterBegin(g2n,nvec,gvec,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd  (g2n,nvec,gvec,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  /* global -> local */
  ierr = VecScatterBegin(g2l,gvec,lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (g2l,gvec,lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  ierr = VecStrideMin(gvec,nsd,NULL,&min_w);CHKERRQ(ierr);
  ierr = VecStrideMax(gvec,nsd,NULL,&max_w);CHKERRQ(ierr);
  iga->rational = ((max_w-min_w)>tol_w) ? PETSC_TRUE : PETSC_FALSE;

  {
    PetscInt n,a,i,pos;
    PetscReal *X,*W;
    const PetscScalar *Xw;
    PetscInt *gwidth = iga->geom_gwidth,gsize = gwidth[0]*gwidth[1]*gwidth[2];
    ierr = VecGetSize(lvec,&n);CHKERRQ(ierr);
    n /= (nsd+1);
    if (n != gsize) {ierr = PetscFree(iga->geometryX);CHKERRQ(ierr);}
    if (n != gsize) {ierr = PetscFree(iga->rationalW);CHKERRQ(ierr);}
    if (!iga->geometryX) {ierr = PetscMalloc1((size_t)(n*nsd),&iga->geometryX);CHKERRQ(ierr);}
    if (!iga->rationalW) {ierr = PetscMalloc1((size_t)n,&iga->rationalW);CHKERRQ(ierr);}
    X = iga->geometryX; W = iga->rationalW;
    ierr = VecGetArrayRead(lvec,&Xw);CHKERRQ(ierr);
    for (pos=0,a=0; a<n; a++) {
      for (i=0; i<nsd; i++)
        X[i+a*nsd] = PetscRealPart(Xw[pos++]);
      W[a] = PetscRealPart(Xw[pos++]);
      if (PetscAbsReal(W[a]) > 0)
        for (i=0; i<nsd; i++)
          X[i+a*nsd] /= W[a];
    }
    ierr = VecRestoreArrayRead(lvec,&Xw);CHKERRQ(ierr);
  }

  ierr = VecScatterDestroy(&g2n);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&g2l);CHKERRQ(ierr);
  ierr = VecDestroy(&lvec);CHKERRQ(ierr);
  ierr = VecDestroy(&gvec);CHKERRQ(ierr);
  ierr = VecDestroy(&nvec);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode IGASaveGeometry(IGA iga,PetscViewer viewer)
{
  PetscBool      isbinary;
  PetscBool      skipheader;
  PetscInt       nsd;
  Vec            nvec,gvec,lvec;
  VecScatter     l2g,g2n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(iga,1,viewer,2);
  IGACheckSetUpStage1(iga,1);
  if (!iga->geometry || !iga->geometryX) SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_ARG_WRONGSTATE,"No geometry set");

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (!isbinary) SETERRQ(((PetscObject)viewer)->comm,PETSC_ERR_ARG_WRONG,"Only for binary viewers");
  ierr = PetscViewerBinaryGetSkipHeader(viewer,&skipheader);CHKERRQ(ierr);

  ierr = IGAGetGeometryDim(iga,&nsd);CHKERRQ(ierr);
  {
    IGA_Grid grid;
    ierr = IGA_NewGridIO(iga,nsd+1,&grid);CHKERRQ(ierr);
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
  {
    PetscInt n,a,i,pos;
    PetscScalar *Xw;
    const PetscReal *X = iga->geometryX;
    const PetscReal *W = iga->rationalW;
    ierr = VecGetSize(lvec,&n);CHKERRQ(ierr);
    n /= (nsd+1);
    ierr = VecGetArray(lvec,&Xw);CHKERRQ(ierr);
    for (pos=0,a=0; a<n; a++) {
      PetscReal w = (W && PetscAbsReal(W[a]) > 0) ? W[a] : 1;
      for (i=0; i<nsd; i++)
        Xw[pos++] = X[i+a*nsd] * w;
      Xw[pos++] = W ? W[a] : (PetscReal)1;
    }
    ierr = VecRestoreArray(lvec,&Xw);CHKERRQ(ierr);
  }
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

/*@
   IGASetPropertyDim - Sets the dimension of the property

   Logically Collective on IGA

   Input Parameters:
+  iga - the IGA context
-  dim - the dimension of the property

   Level: normal

.keywords: IGA, dimension
@*/
PetscErrorCode IGASetPropertyDim(IGA iga,PetscInt dim)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidLogicalCollectiveInt(iga,dim,2);
  if (dim < 0) SETERRQ1(((PetscObject)iga)->comm,PETSC_ERR_ARG_OUTOFRANGE,
                        "Number of properties must be nonnegative, got %D",dim);
  if (iga->property == dim) PetscFunctionReturn(0);
  iga->property = dim;
  ierr = PetscFree(iga->propertyA);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGAGetPropertyDim(IGA iga,PetscInt *dim)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(dim,2);
  *dim = iga->property;
  PetscFunctionReturn(0);
}

PetscErrorCode IGALoadProperty(IGA iga,PetscViewer viewer)
{
  PetscBool      isbinary;
  PetscBool      skipheader;
  PetscInt       npd;
  Vec            nvec,gvec,lvec;
  VecScatter     g2n,g2l;

  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(iga,1,viewer,2);
  IGACheckSetUpStage1(iga,1);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (!isbinary) SETERRQ(((PetscObject)viewer)->comm,PETSC_ERR_ARG_WRONG,"Only for binary viewers");
  ierr = PetscViewerBinaryGetSkipHeader(viewer,&skipheader);CHKERRQ(ierr);

  ierr = IGAGetPropertyDim(iga,&npd);CHKERRQ(ierr);
  if (npd < 1) SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_ARG_WRONGSTATE,
                       "Must call IGASetPropertyDim() first");
  {
    IGA_Grid grid;
    ierr = IGA_NewGridIO(iga,npd,&grid);CHKERRQ(ierr);
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
  /* viewer -> natural*/
  if (!skipheader) {
    ierr = VecLoad(nvec,viewer);CHKERRQ(ierr);
  } else {
    ierr = VecLoad_Binary_SkipHeader(nvec,viewer);CHKERRQ(ierr);
  }
  /* natural -> global */
  ierr = VecScatterBegin(g2n,nvec,gvec,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd  (g2n,nvec,gvec,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  /* global -> local */
  ierr = VecScatterBegin(g2l,gvec,lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (g2l,gvec,lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  ierr = PetscFree(iga->propertyA);CHKERRQ(ierr);
  {
    PetscInt n; const PetscScalar *A;
    ierr = VecGetSize(lvec,&n);CHKERRQ(ierr);
    ierr = PetscMalloc1((size_t)n,&iga->propertyA);CHKERRQ(ierr);
    ierr = VecGetArrayRead(lvec,&A);CHKERRQ(ierr);
    ierr = PetscMemcpy(iga->propertyA,A,(size_t)n*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(lvec,&A);CHKERRQ(ierr);
  }

  ierr = VecScatterDestroy(&g2n);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&g2l);CHKERRQ(ierr);
  ierr = VecDestroy(&lvec);CHKERRQ(ierr);
  ierr = VecDestroy(&gvec);CHKERRQ(ierr);
  ierr = VecDestroy(&nvec);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode IGASaveProperty(IGA iga,PetscViewer viewer)
{
  PetscBool      isbinary;
  PetscBool      skipheader;
  PetscInt       npd;
  Vec            nvec,gvec,lvec;
  VecScatter     l2g,g2n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(iga,1,viewer,2);
  IGACheckSetUpStage1(iga,1);
  if (!iga->property) SETERRQ(((PetscObject)iga)->comm,PETSC_ERR_ARG_WRONGSTATE,"No property set");

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (!isbinary) SETERRQ(((PetscObject)viewer)->comm,PETSC_ERR_ARG_WRONG,"Only for binary viewers");
  ierr = PetscViewerBinaryGetSkipHeader(viewer,&skipheader);CHKERRQ(ierr);

  ierr = IGAGetPropertyDim(iga,&npd);CHKERRQ(ierr);
  {
    IGA_Grid grid;
    ierr = IGA_NewGridIO(iga,npd,&grid);CHKERRQ(ierr);
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
  {
    PetscInt n; PetscScalar *A;
    ierr = VecGetSize(lvec,&n);CHKERRQ(ierr);
    ierr = VecGetArray(lvec,&A);CHKERRQ(ierr);
    ierr = PetscMemcpy(A,iga->propertyA,(size_t)n*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = VecRestoreArray(lvec,&A);CHKERRQ(ierr);
  }
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

#if PETSC_VERSION_LT(3,12,0)
#define PetscBinaryRead(fd,data,num,count,type) PetscBinaryRead(fd,data,num,type)
#endif

static PetscErrorCode VecLoad_Binary_SkipHeader(Vec vec,PetscViewer viewer)
{
  MPI_Comm       comm;
  PetscMPIInt    i,rank,size,tag,count;
  int            fd;
  PetscInt       n;
  const PetscInt *range;
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
    ierr = PetscBinaryRead(fd,array,n,NULL,PETSC_SCALAR);CHKERRQ(ierr);
    if (size > 1) {
      ierr = VecGetOwnershipRanges(vec,&range);CHKERRQ(ierr);
      n = 1;
      for (i=1; i<size; i++)
        n = PetscMax(n,range[i+1] - range[i]);
      ierr = PetscMalloc1((size_t)n,&work);CHKERRQ(ierr);
      for (i=1; i<size; i++) {
        n = range[i+1] - range[i];
        ierr = PetscMPIIntCast(n,&count);CHKERRQ(ierr);
        ierr = PetscBinaryRead(fd,work,count,NULL,PETSC_SCALAR);CHKERRQ(ierr);
        ierr = MPI_Send(work,count,MPIU_SCALAR,i,tag,comm);CHKERRQ(ierr);
      }
      ierr = PetscFree(work);CHKERRQ(ierr);
    }
  } else {
    MPI_Status status;
    ierr = PetscMPIIntCast(n,&count);CHKERRQ(ierr);
    ierr = MPI_Recv(array,count,MPIU_SCALAR,0,tag,comm,&status);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(vec,&array);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

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
  IGACheckSetUpStage2(iga,1);

  ierr = IGAGetNaturalVec(iga,&natural);CHKERRQ(ierr);
  ierr = VecLoad(natural,viewer);CHKERRQ(ierr);
  ierr = IGANaturalToGlobal(iga,natural,vec);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

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
  IGACheckSetUpStage2(iga,1);

  ierr = IGAGetNaturalVec(iga,&natural);CHKERRQ(ierr);
  ierr = IGAGlobalToNatural(iga,vec,natural);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)natural,((PetscObject)vec)->name);CHKERRQ(ierr);
  ierr = VecView(natural,viewer);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

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
  IGACheckSetUpStage2(iga,1);

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
  IGACheckSetUpStage2(iga,1);

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
