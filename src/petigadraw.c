#include "petiga.h"

PETSC_EXTERN PetscErrorCode IGACreateDrawDM(IGA iga,PetscInt bs,const PetscInt res[],DM *dm);
PETSC_EXTERN PetscErrorCode IGAGetDrawDM(IGA iga,DM *dm);

EXTERN_C_BEGIN
extern PetscReal IGA_Greville(PetscInt i,PetscInt p,const PetscReal U[]);
EXTERN_C_END

static
PetscReal GrevillePoint(PetscInt index,PETSC_UNUSED PetscInt step,const IGAAxis axis)
{
  if (PetscUnlikely(axis->p == 0)) return 0;
  return IGA_Greville(index,axis->p,axis->U);
}

static
PetscReal LagrangePoint(PetscInt index,PetscInt step,const IGAAxis axis)
{
  if (PetscUnlikely(axis->p == 0)) return 0;
  {
    PetscInt n = axis->nel;
    PetscInt p = step;
    PetscInt e = index / p;
    PetscInt i = index % p;
    PetscReal u0,u1;
    if (PetscUnlikely(e == n)) { e -= 1; i = p; }
    u0 = axis->U[axis->span[e]];
    u1 = axis->U[axis->span[e]+1];
    return u0 + (PetscReal)i/(PetscReal)p*(u1-u0);
  }
}

#if PETSC_VERSION_LT(3,8,0)
#define PETSCVIEWERGLVIS "glvis"
#endif

PetscErrorCode IGACreateDrawDM(IGA iga,PetscInt bs,const PetscInt res[],DM *dm)
{
  MPI_Comm        comm;
  PetscInt        i,dim,nsd;
  PetscInt        sizes[3] = {1, 1, 1};
  PetscInt        width[3] = {1, 1, 1};
  PetscBool       wraps[3] = {PETSC_TRUE, PETSC_TRUE, PETSC_TRUE};
  PetscInt        resolution[3] = {1,1,1};
  PetscInt        n,N;
  Vec             X;
  PetscScalar     *arrayX;
  IGAProbe        probe;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidLogicalCollectiveInt(iga,bs,2);
  if (res) PetscValidIntPointer(res,3);
  PetscValidPointer(dm,2);
  IGACheckSetUpStage1(iga,1);

  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
  dim = PetscClipInterval(dim,1,3); /* silent GCC -O3 warning */
  /* determine resolution */
  for (i=0; i<dim; i++) resolution[i] = (res && res[i] != PETSC_DECIDE) ? res[i] : iga->axis[i]->p;
  for (i=0; i<dim; i++) resolution[i] = PetscMax(resolution[i],1);
  /* compute global and local sizes */
  if (!iga->collocation) {
    const PetscInt *pranks = iga->proc_ranks;
    const PetscInt *psizes = iga->proc_sizes;
    for (i=0; i<dim; i++) {
      sizes[i] = iga->elem_sizes[i]*resolution[i] + 1;
      width[i] = iga->elem_width[i]*resolution[i] + (pranks[i] == psizes[i]-1);
    }
  } else {
    for (i=0; i<dim; i++) {
      sizes[i] = iga->node_sizes[i];
      width[i] = iga->node_lwidth[i];
    }
  }
  /* create DMDA context */
  ierr = IGACreateDMDA(iga,bs,sizes,width,wraps,PETSC_TRUE,1,dm);CHKERRQ(ierr);
  /* create coordinate vector */
  ierr = IGAGetGeometryDim(iga,&nsd);CHKERRQ(ierr);
  nsd = PetscClipInterval(nsd,dim,3);
  n = width[0]*width[1]*width[2];
  N = sizes[0]*sizes[1]*sizes[2];
  ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
  ierr = VecCreate(comm,&X);CHKERRQ(ierr);
  ierr = VecSetSizes(X,nsd*n,nsd*N);CHKERRQ(ierr);
  ierr = VecSetBlockSize(X,nsd);CHKERRQ(ierr);
  ierr = VecSetType(X,VECSTANDARD);CHKERRQ(ierr);
  ierr = VecSetUp(X);CHKERRQ(ierr);
  ierr = DMSetCoordinates(*dm,X);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  /* fill coordinate vector */
  ierr = DMGetCoordinates(*dm,&X);CHKERRQ(ierr);
  ierr = VecGetArray(X,&arrayX);CHKERRQ(ierr);
  ierr = IGAProbeCreate(iga,NULL,&probe);CHKERRQ(ierr);
  ierr = IGAProbeSetOrder(probe,0);CHKERRQ(ierr);
  ierr = IGAProbeSetCollective(probe,PETSC_FALSE);CHKERRQ(ierr);
  {
    PetscReal uvw[3]  = {0,0,0};
    PetscReal xval[3] = {0,0,0};
    PetscInt is,iw,js,jw,ks,kw;
    PetscInt c,j,k,xpos=0;
    const PetscInt *shift = iga->node_shift;
    const IGAAxis  *axis  = iga->axis;
    PetscReal (*ComputePoint)(PetscInt,PetscInt,IGAAxis);
    ComputePoint = iga->collocation ? GrevillePoint : LagrangePoint;
    ierr = DMDAGetCorners(*dm,&is,&js,&ks,&iw,&jw,&kw);CHKERRQ(ierr);
    for (k=ks; k<ks+kw; k++) {
      uvw[2] = ComputePoint(k+shift[2],resolution[2],axis[2]);
      for (j=js; j<js+jw; j++) {
        uvw[1] = ComputePoint(j+shift[1],resolution[1],axis[1]);
        for (i=is; i<is+iw; i++) {
          uvw[0] = ComputePoint(i+shift[0],resolution[0],axis[0]);
          {
            ierr = IGAProbeSetPoint(probe,uvw);CHKERRQ(ierr);
            ierr = IGAProbeGeomMap(probe,xval);CHKERRQ(ierr);
            for (c=0; c<nsd; c++) arrayX[xpos++] = xval[c];
          }
        }
      }
    }
  }
  ierr = IGAProbeDestroy(&probe);CHKERRQ(ierr);
  ierr = VecRestoreArray(X,&arrayX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGAGetDrawDM(IGA iga,DM *dm)
{
  const char     *prefix;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(dm,2);
  IGACheckSetUpStage2(iga,1);
  if (!iga->draw_dm) {
    /* determine resolution */
    PetscInt i,dim,num,resolution[3] = {1,1,1};
    ierr = IGAGetOptionsPrefix(iga,&prefix);CHKERRQ(ierr);
    ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
    dim = num = PetscClipInterval(dim,1,3);
    for (i=0; i<dim; i++) resolution[i] = iga->axis[i]->p;
    ierr = PetscOptionsGetIntArray(((PetscObject)iga)->options,prefix,"-iga_draw_resolution",resolution,&num,NULL);CHKERRQ(ierr);
    if (num == 1) for (i=1; i<dim; i++) resolution[i] = resolution[0];
    /* create DMDA draw context */
    ierr = IGACreateDrawDM(iga,iga->dof,resolution,&iga->draw_dm);CHKERRQ(ierr);
    if (iga->fieldname)
      for (i=0; i<iga->dof; i++)
        {ierr = DMDASetFieldName(iga->draw_dm,i,iga->fieldname[i]);CHKERRQ(ierr);}
  }
  *dm = iga->draw_dm;
  PetscFunctionReturn(0);
}

PetscErrorCode IGADrawVec(IGA iga,Vec vec,PetscViewer viewer)
{
  DM             dm;
  PetscInt       dof;
  PetscInt       resolution[3] = {1,1,1},d,dim,M[3];
  Vec            U;
  PetscScalar    *arrayU=NULL;
  IGAProbe       probe;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,3);
  PetscCheckSameComm(iga,1,vec,2);
  PetscCheckSameComm(iga,1,viewer,3);
  IGACheckSetUp(iga,1);

  ierr = IGAGetDrawDM(iga,&dm);CHKERRQ(ierr);
  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
  dim = PetscClipInterval(dim,1,3);
  ierr = DMDAGetInfo(dm,NULL,&M[0],&M[1],&M[2],NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  for (d=0; d<dim; d++) resolution[d] = (M[d]-1)/iga->elem_sizes[d];

  ierr = DMGetGlobalVector(dm,&U);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)U,((PetscObject)vec)->name);CHKERRQ(ierr);
  ierr = IGAGetDof(iga,&dof);CHKERRQ(ierr);
  ierr = VecGetArray(U,&arrayU);CHKERRQ(ierr);
  ierr = IGAProbeCreate(iga,vec,&probe);CHKERRQ(ierr);
  ierr = IGAProbeSetOrder(probe,0);CHKERRQ(ierr);
  ierr = IGAProbeSetCollective(probe,PETSC_FALSE);CHKERRQ(ierr);
  {
    PetscReal uvw[3]  = {0,0,0};
    PetscScalar *uval;
    PetscInt is,iw,js,jw,ks,kw;
    PetscInt c,i,j,k,upos=0;
    const PetscInt *shift = iga->node_shift;
    const IGAAxis  *axis  = iga->axis;
    PetscReal (*ComputePoint)(PetscInt,PetscInt,IGAAxis);
    ComputePoint = iga->collocation ? GrevillePoint : LagrangePoint;
    ierr = DMDAGetCorners(dm,&is,&js,&ks,&iw,&jw,&kw);CHKERRQ(ierr);
    ierr = PetscMalloc1((size_t)dof,&uval);CHKERRQ(ierr);
    for (k=ks; k<ks+kw; k++) {
      uvw[2] = ComputePoint(k+shift[2],resolution[2],axis[2]);
      for (j=js; j<js+jw; j++) {
        uvw[1] = ComputePoint(j+shift[1],resolution[1],axis[1]);
        for (i=is; i<is+iw; i++) {
          uvw[0] = ComputePoint(i+shift[0],resolution[0],axis[0]);
          {
            ierr = IGAProbeSetPoint(probe,uvw);CHKERRQ(ierr);
            ierr = IGAProbeFormValue(probe,uval);CHKERRQ(ierr);
            for (c=0; c<dof; c++) arrayU[upos++] = uval[c];
          }
        }
      }
    }
    ierr = PetscFree(uval);CHKERRQ(ierr);
  }
  ierr = IGAProbeDestroy(&probe);CHKERRQ(ierr);
  ierr = VecRestoreArray(U,&arrayU);CHKERRQ(ierr);
  ierr = VecView(U,viewer);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)U,NULL);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm,&U);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGADrawVecVTK(IGA iga,Vec vec,const char filename[])
{
  MPI_Comm       comm;
  PetscViewer    viewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,1);
  PetscCheckSameComm(iga,1,vec,2);
  PetscValidCharPointer(filename,2);

  ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
  ierr = PetscViewerVTKOpen(comm,filename,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = IGADrawVec(iga,vec,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
extern PetscInt IGA_NextKnot(PetscInt m,const PetscReal U[],PetscInt k,PetscInt direction);
EXTERN_C_END

PetscErrorCode IGADraw(IGA iga,PetscViewer viewer)
{
  DM                dm;
  PetscBool         match;
  PetscViewerFormat format;
  PetscInt          i,j,dim;
  PetscDraw         draw;
  MPI_Comm          comm;
  PetscMPIInt       size,rank;
  PetscReal         xmin=0,xmax=0,xlen=0,xb=0;
  PetscReal         ymin=0,ymax=0,ylen=0,yb=0;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(iga,1,viewer,2);
  IGACheckSetUp(iga,1);

  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&match);CHKERRQ(ierr);
  if (match && format == PETSC_VIEWER_ASCII_VTK) {
    ierr = IGAGetDrawDM(iga,&dm);CHKERRQ(ierr);
    ierr = DMView(dm,viewer);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERGLVIS,&match);CHKERRQ(ierr);
  if (match) {
    ierr = IGAGetDrawDM(iga,&dm);CHKERRQ(ierr);
    ierr = DMView(dm,viewer);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERVTK,&match);CHKERRQ(ierr);
  if (match) {
    ierr = IGAGetDrawDM(iga,&dm);CHKERRQ(ierr);
    ierr = DMView(dm,viewer);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&match);CHKERRQ(ierr);
  if (!match) PetscFunctionReturn(0);

  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
  if (dim != 2) PetscFunctionReturn(0);

  ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawIsNull(draw,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  ierr = PetscDrawCheckResizedWindow(draw);CHKERRQ(ierr);
  ierr = PetscDrawClear(draw);CHKERRQ(ierr);

  for (i=0; i<2; i++) {
    IGAAxis axis; PetscReal Ua,Ub;
    ierr = IGAGetAxis(iga,i,&axis);CHKERRQ(ierr);
    ierr = IGAAxisGetLimits(axis,&Ua,&Ub);CHKERRQ(ierr);
    if (!i) { xmin = Ua; xmax = Ub; xlen = xmax-xmin; xb = (PetscReal)0.05*xlen;}
    else    { ymin = Ua; ymax = Ub; ylen = ymax-ymin; yb = (PetscReal)0.05*ylen;}
  }
  ierr = PetscDrawSetCoordinates(draw,xmin-xb,ymin-yb,xmax+xb,ymax+yb);CHKERRQ(ierr);

  if (size > 1) { /* Processor grid */
    PetscInt *r = iga->proc_ranks;
    int colors[] = {PETSC_DRAW_GRAY, PETSC_DRAW_WHITE};
    int c = colors[(r[0]+r[1]) % 2]; /* chessboard */
    PetscReal x0=0,y0=0,x=0,y=0;
    for (i=0; i<2; i++) {
      PetscReal Ua,Ub;
      if (!iga->collocation) {
        PetscInt a = iga->elem_start[i];
        PetscInt b = a + iga->elem_width[i] - 1;
        IGAAxis  axis; PetscInt *span; PetscReal *U;
        ierr = IGAGetAxis(iga,i,&axis);CHKERRQ(ierr);
        ierr = IGAAxisGetSpans(axis,NULL,&span);CHKERRQ(ierr);
        ierr = IGAAxisGetKnots(axis,NULL,&U);CHKERRQ(ierr);
        Ua = U[span[a]];
        Ub = U[span[b]+1];
      } else {
        IGABasis BD = iga->basis[i];
        PetscInt a = iga->node_lstart[i] ;
        PetscInt b = a + iga->node_lwidth[i] - 1;
        PetscInt n = iga->node_sizes[i] - 1;
        PetscReal *U = BD->point;
        Ua = U[a] + ((a>0) ? (U[a-1]-U[a])/2 : 0);
        Ub = U[b] + ((b<n) ? (U[b+1]-U[b])/2 : 0);
      }
      if (!i) { x0 = Ua; x = Ub;}
      else    { y0 = Ua; y = Ub;}
    }
    ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
    ierr = PetscDrawRectangle(draw,x0,y0,x,y,c,c,c,c);CHKERRQ(ierr);
    ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
    ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
  }

  ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
  if (!rank) { /* Element grid */
    int c = PETSC_DRAW_BLACK;
    IGAAxis axis;
    PetscInt p,m;
    PetscReal *U;
    ierr = IGAGetAxis(iga,0,&axis);CHKERRQ(ierr);
    ierr = IGAAxisGetDegree(axis,&p);CHKERRQ(ierr);
    ierr = IGAAxisGetKnots(axis,&m,&U);CHKERRQ(ierr);
    for (i=p; i<=m-p; i = IGA_NextKnot(m,U,i,1)) {
      PetscReal x = U[i];
      ierr = PetscDrawLine(draw,x,ymin,x,ymax,c);CHKERRQ(ierr);
    }
    ierr = IGAGetAxis(iga,1,&axis);CHKERRQ(ierr);
    ierr = IGAAxisGetDegree(axis,&p);CHKERRQ(ierr);
    ierr = IGAAxisGetKnots(axis,&m,&U);CHKERRQ(ierr);
    for (i=p; i<=m-p; i = IGA_NextKnot(m,U,i,1)) {
      PetscReal y = U[i];
      ierr = PetscDrawLine(draw,xmin,y,xmax,y,c);CHKERRQ(ierr);
    }
  }
  ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
  ierr = PetscDrawFlush(draw);CHKERRQ(ierr);

  ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
  if (!rank) { /* Quadrature/Collocation points */
    int c = PETSC_DRAW_RED;
    IGABasis *BD = iga->basis;
    PetscInt  ni = BD[0]->nel*BD[0]->nqp;
    PetscInt  nj = BD[1]->nel*BD[1]->nqp;
    PetscReal *u = BD[0]->point, *wu = BD[0]->weight;
    PetscReal *v = BD[1]->point, *wv = BD[1]->weight;
    for (i=0; i<ni; i++) {
      PetscReal x = u[i]; if (wu[i] <= 0) continue;
      for (j=0; j<nj; j++) {
        PetscReal y = v[j]; if (wv[j] <= 0) continue;
        ierr = PetscDrawPoint(draw,x,y,c);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
  ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
  ierr = PetscDrawPause(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
