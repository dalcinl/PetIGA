#include "petiga.h"

#if PETSC_VERSION_LT(3,5,0)
#define PCGetOperators(pc,A,B) PCGetOperators(pc,A,B,NULL)
#endif

/* ---------------------------------------------------------------- */

PETSC_STATIC_INLINE
PetscInt Index(const PetscInt N[],PetscInt i,PetscInt j,PetscInt k)
{
  return i + j * N[0] + k * N[0] * N[1];
}

PETSC_STATIC_INLINE
PetscBool OnGrid(const PetscBool W[3],const PetscInt N[3],
                 PetscInt *i,PetscInt *j,PetscInt *k)
{
  if (W[0]) { if (*i<0) *i += N[0]; else if (*i>=N[0]) *i -= N[0]; }
  if (W[1]) { if (*j<0) *j += N[1]; else if (*j>=N[1]) *j -= N[1]; }
  if (W[2]) { if (*k<0) *k += N[2]; else if (*k>=N[2]) *k -= N[2]; }
  if (*i<0 || *i>=N[0]) return PETSC_FALSE;
  if (*j<0 || *j>=N[1]) return PETSC_FALSE;
  if (*k<0 || *k>=N[2]) return PETSC_FALSE;
  return PETSC_TRUE;
}

PETSC_STATIC_INLINE
PetscInt Color(const PetscInt shape[3],
               const PetscInt start[3],
               const PetscInt width[3],
               PetscInt i,PetscInt j,PetscInt k)
{
  PetscInt L[3],R[3],C[2],r=0,g=0,b=0;
  L[0] = start[0]; R[0] = start[0] + width[0] - 1;
  L[1] = start[1]; R[1] = start[1] + width[1] - 1;
  L[2] = start[2]; R[2] = start[2] + width[2] - 1;
  if (i<L[0]) r = i - L[0]; if (i>R[0]) r = i - R[0];
  if (j<L[1]) g = j - L[1]; if (j>R[1]) g = j - R[1];
  if (k<L[2]) b = k - L[2]; if (k>R[2]) b = k - R[2];
  C[0] = shape[0] - width[0] + 1;
  C[1] = shape[1] - width[1] + 1;
  return Index(C,r,g,b);
}

static const
PetscInt STENCIL[7][3] = {{ 0,  0, -1},
                          { 0, -1,  0},
                          {-1,  0,  0},
                          { 0,  0,  0},
                          {+1,  0,  0},
                          { 0, +1,  0},
                          { 0,  0, +1}};

static
#undef  __FUNCT__
#define __FUNCT__ "IGAComputeBDDCGraph"
PetscErrorCode IGAComputeBDDCGraph(PetscInt bs,
                                   const PetscBool wrap[3],const PetscInt shape[3],
                                   const PetscInt start[3],const PetscInt width[3],
                                   PetscInt *_nvtx,PetscInt *_xadj[],PetscInt *_adjy[])
{
  PetscInt       c,i,j,k,s,v,pos;
  PetscInt       nvtx=0,*xadj;
  PetscInt       nadj=0,*adjy;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidIntPointer(_nvtx,6);
  PetscValidPointer(_xadj,7);
  PetscValidPointer(_adjy,8);

  /* Compute the number of vertices and adjacencies */
  nvtx = shape[0]*shape[1]*shape[2];
  for (k=0; k<shape[2]; k++)
    for (j=0; j<shape[1]; j++)
      for (i=0; i<shape[0]; i++) {
        PetscInt color = Color(shape,start,width,i,j,k);
        for (s=0; s<7; s++) {
          PetscInt ii = i + STENCIL[s][0];
          PetscInt jj = j + STENCIL[s][1];
          PetscInt kk = k + STENCIL[s][2];
          if (OnGrid(wrap,shape,&ii,&jj,&kk)) {
            PetscInt cc = Color(shape,start,width,ii,jj,kk);
            if (cc == color) nadj++;
          }
        }
      }

  /* Allocate arrays to store the adjacency graph */
  nvtx *= bs; nadj *= bs; /* adjust for block size */
  ierr = PetscMalloc((nvtx+1)*sizeof(PetscInt),&xadj);CHKERRQ(ierr);
  ierr = PetscMalloc(nadj*sizeof(PetscInt),&adjy);CHKERRQ(ierr);

  /* Fill the adjacency graph */
  pos = 0; xadj[pos++] = 0;
  for (k=0; k<shape[2]; k++)
    for (j=0; j<shape[1]; j++)
      for (i=0; i<shape[0]; i++) {
        /* Compute the color of this vertex */
        PetscInt color = Color(shape,start,width,i,j,k);
        /* Compute the list of neighbor vertices
           having the same color */
        PetscInt nv = 0, vertices[7];
        for (s=0; s<7; s++) { /* loop over neighbors */
          PetscInt ii = i + STENCIL[s][0];
          PetscInt jj = j + STENCIL[s][1];
          PetscInt kk = k + STENCIL[s][2];
          if (OnGrid(wrap,shape,&ii,&jj,&kk)) {
            PetscInt cc = Color(shape,start,width,ii,jj,kk);
            if (cc == color)
              vertices[nv++] = Index(shape,ii,jj,kk);
          }
        }
        ierr = PetscSortInt(nv,vertices);CHKERRQ(ierr);
        for (c=0; c<bs; c++) {
          xadj[pos] = xadj[pos-1];
          for (v=0; v<nv; v++)
            adjy[xadj[pos]++] = c + bs*vertices[v];
          pos++;
        }
      }

  *_nvtx = nvtx;
  *_xadj = xadj;
  *_adjy = adjy;

  PetscFunctionReturn(0);
}

#if PETSC_VERSION_LE(3,3,0)
#include "petscbt.h"
PETSC_STATIC_INLINE char PetscBTLookupClear(PetscBT array,PetscInt index)
{
  char      BT_mask,BT_c;
  PetscInt  BT_idx;
  return (BT_idx        = (index)/PETSC_BITS_PER_BYTE,
          BT_c          = array[BT_idx],
          BT_mask       = (char)1 << ((index)%PETSC_BITS_PER_BYTE),
          array[BT_idx] = BT_c & (~BT_mask),
          BT_c & BT_mask);
}
#endif

static
#undef  __FUNCT__
#define __FUNCT__ "IGAComputeBDDCBoundary"
PetscErrorCode IGAComputeBDDCBoundary(PetscInt dim,PetscInt bs,const PetscInt shape[3],
                                      PetscBool atbnd[][2],PetscInt count[][2],PetscInt *field[][2],
                                      PetscInt *_ndirichlet,PetscInt *_idirichlet[],
                                      PetscInt *_nneumann,  PetscInt *_ineumann[])
{
  PetscBT        Dmask,Nmask;
  PetscInt       i, m = bs*shape[0]*shape[1]*shape[2];
  PetscInt       dir, side, ijk[3], index = 0, pos;
  PetscInt       ndirichlet = 0, *idirichlet = NULL;
  PetscInt       nneumann   = 0, *ineumann   = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidIntPointer(_ndirichlet,7);
  PetscValidPointer(_idirichlet,8);
  PetscValidIntPointer(_nneumann,9);
  PetscValidPointer(_ineumann,10);

  ierr = PetscBTCreate(m,&Dmask);CHKERRQ(ierr);
  ierr = PetscBTCreate(m,&Nmask);CHKERRQ(ierr);

  for (ijk[2]=0; ijk[2]<shape[2]; ijk[2]++)
    for (ijk[1]=0; ijk[1]<shape[1]; ijk[1]++)
      for (ijk[0]=0; ijk[0]<shape[0]; ijk[0]++, index++)
        for (dir=0; dir<dim; dir++)
          for (side=0; side<=1; side++)
            if (atbnd[dir][side] && ijk[dir] == (!side?0:shape[dir]-1))
              {
                PetscInt c,n = count[dir][side];
                PetscInt  *f = field[dir][side];
                for (c=0; c<n; c++) {
                  i = f[c] + bs*index;
                  if (!PetscBTLookupSet(Dmask,i))  ndirichlet++;
                  if (PetscBTLookupClear(Nmask,i)) nneumann--;
                }
                for (c=0; c<bs; c++) {
                  i = c + bs*index;
                  if (!PetscBTLookup(Dmask,i))
                    if (!PetscBTLookupSet(Nmask,i)) nneumann++;
                }
              }

  ierr = PetscMalloc(ndirichlet*sizeof(PetscInt),&idirichlet);CHKERRQ(ierr);
  for (pos=0,i=0; i<m; i++) if (PetscBTLookup(Dmask,i)) idirichlet[pos++] = i;
  ierr = PetscBTDestroy(&Dmask);CHKERRQ(ierr);
  *_ndirichlet = ndirichlet;
  *_idirichlet = idirichlet;

  ierr = PetscMalloc(nneumann*sizeof(PetscInt),&ineumann);CHKERRQ(ierr);
  for (pos=0,i=0; i<m; i++) if (PetscBTLookup(Nmask,i)) ineumann[pos++] = i;
  ierr = PetscBTDestroy(&Nmask);CHKERRQ(ierr);
  *_nneumann = nneumann;
  *_ineumann = ineumann;

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAPreparePCBDDC"
PetscErrorCode IGAPreparePCBDDC(IGA iga,PC pc)
{
  Mat            mat;
  void           (*f)(void);
  const char     *prefix;
  PetscBool      graph = PETSC_TRUE;
  PetscBool      boundary = PETSC_TRUE;
  PetscBool      nullspace = PETSC_TRUE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(pc,PC_CLASSID,2);
  IGACheckSetUpStage2(iga,1);

  {
    Mat A,B;
    PetscBool useAmat = PETSC_FALSE;
    ierr = PCGetOperators(pc,&A,&B);CHKERRQ(ierr);
#if PETSC_VERSION_GE(3,4,0)
    ierr = PCGetUseAmat(pc,&useAmat);CHKERRQ(ierr);
#endif
    mat = useAmat ? A : B;
  }

  ierr = PetscObjectQueryFunction((PetscObject)mat,"MatISGetLocalMat_C",&f);CHKERRQ(ierr);
  if (!f) PetscFunctionReturn(0);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCBDDCSetLocalAdjacencyGraph_C",&f);CHKERRQ(ierr);
  if (!f) PetscFunctionReturn(0);

  ierr = IGAGetOptionsPrefix(iga,&prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(prefix,"-iga_set_bddc_graph",&graph,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(prefix,"-iga_set_bddc_boundary",&boundary,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(prefix,"-iga_set_bddc_nullspace",&nullspace,NULL);CHKERRQ(ierr);

  if (graph) {
    PetscInt i,dim,dof;
    PetscBool wrap[3] = {PETSC_FALSE,PETSC_FALSE,PETSC_FALSE};
    PetscInt shape[3] = {1,1,1};
    PetscInt start[3] = {0,0,0};
    PetscInt width[3] = {1,1,1};
    PetscInt nvtx=0,*xadj=NULL,*adjy=NULL;
    ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
    ierr = IGAGetDof(iga,&dof);CHKERRQ(ierr);
    for (i=0; i<dim; i++) {
      shape[i] = iga->node_gwidth[i];
      if (iga->proc_sizes[i] > 1) {
        start[i] = iga->node_lstart[i] - iga->node_gstart[i];
        width[i] = iga->node_lwidth[i];
      } else {
        start[i] = iga->node_gstart[i];
        width[i] = iga->node_gwidth[i];
        wrap[i]  = iga->axis[i]->periodic;
      }
    }
    ierr = IGAComputeBDDCGraph(dof,wrap,shape,start,width,&nvtx,&xadj,&adjy);CHKERRQ(ierr);
#if defined(PETSC_HAVE_PCBDDC) /* XXX */
    ierr = PCBDDCSetLocalAdjacencyGraph(pc,nvtx,xadj,adjy,PETSC_OWN_POINTER);CHKERRQ(ierr);
#else
    ierr = PetscFree(xadj);CHKERRQ(ierr);
    ierr = PetscFree(adjy);CHKERRQ(ierr);
#endif
  }

  if (boundary) {
    PetscInt  i,s,dim,dof;
    PetscInt  shape[3]    = {1,1,1};
    PetscBool atbnd[3][2] = {{PETSC_FALSE,PETSC_FALSE},
                             {PETSC_FALSE,PETSC_FALSE},
                             {PETSC_FALSE,PETSC_FALSE}};
    PetscInt  count[3][2] = {{0,0},{0,0},{0,0}};
    PetscInt *field[3][2] = {{0,0},{0,0},{0,0}};
    PetscInt  nd=0,*id=NULL;
    PetscInt  nn=0,*in=NULL;
    MPI_Comm  comm;
    IS        isd,isn;
    ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
    ierr = IGAGetDof(iga,&dof);CHKERRQ(ierr);
    for (i=0; i<dim; i++) {
      shape[i] = iga->node_gwidth[i];
      if (!iga->axis[i]->periodic)
        for (s=0; s<=1; s++)
          if (iga->proc_ranks[i] == ((!s)?0:iga->proc_sizes[i]-1)) {
            atbnd[i][s] = PETSC_TRUE;
            count[i][s] = iga->form->value[i][s]->count;
            field[i][s] = iga->form->value[i][s]->field;
          }
    }
    ierr = IGAComputeBDDCBoundary(dim,dof,shape,atbnd,count,field,&nd,&id,&nn,&in);CHKERRQ(ierr);
#if PETSC_VERSION_LT(3,5,0)
    comm = PETSC_COMM_SELF;
#else
    comm = PetscObjectComm((PetscObject)pc);
#endif
    ierr = ISCreateGeneral(comm,nd,id,PETSC_OWN_POINTER,&isd);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm,nn,in,PETSC_OWN_POINTER,&isn);CHKERRQ(ierr);
#if defined(PETSC_HAVE_PCBDDC) /* XXX */
#if PETSC_VERSION_LT(3,5,0)
    ierr = PCBDDCSetDirichletBoundaries(pc,isd);CHKERRQ(ierr);
    ierr = PCBDDCSetNeumannBoundaries(pc,isn);CHKERRQ(ierr);
#else
    ierr = PCBDDCSetDirichletBoundariesLocal(pc,isd);CHKERRQ(ierr);
    ierr = PCBDDCSetNeumannBoundariesLocal(pc,isn);CHKERRQ(ierr);
#endif
#endif
    ierr = ISDestroy(&isd);CHKERRQ(ierr);
    ierr = ISDestroy(&isn);CHKERRQ(ierr);
  }

  if (nullspace) {
    MatNullSpace nsp;
    ierr = MatGetNullSpace(mat,&nsp);CHKERRQ(ierr);
#if defined(PETSC_HAVE_PCBDDC) /* XXX */
#if PETSC_VERSION_GE(3,4,0)
    if (nsp) {ierr = PCBDDCSetNullSpace(pc,nsp);CHKERRQ(ierr);}
#endif
#endif
  }

  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */


#undef  __FUNCT__
#define __FUNCT__ "IGA_OptionsHandler_PC"
static PetscErrorCode IGA_OptionsHandler_PC(PetscObject obj,void *ctx)
{
  PC             pc = (PC)obj;
  Mat            mat;
  IGA            iga;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (PetscOptionsPublishCount != 1) PetscFunctionReturn(0);
  ierr = PCGetOperators(pc,NULL,&mat);CHKERRQ(ierr);
  if (!mat) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  ierr = PetscObjectQuery((PetscObject)mat,"IGA",(PetscObject*)&iga);CHKERRQ(ierr);
  if (!iga) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  /* */
  ierr = IGAPreparePCBDDC(iga,pc);CHKERRQ(ierr);
  /* */
  PetscFunctionReturn(0);
}
static PetscErrorCode OptHdlDel(PetscObject obj,void *ctx) {return 0;}

#undef  __FUNCT__
#define __FUNCT__ "IGASetOptionsHandlerPC"
PetscErrorCode IGASetOptionsHandlerPC(PC pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscObjectAddOptionsHandler((PetscObject)pc,IGA_OptionsHandler_PC,OptHdlDel,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
