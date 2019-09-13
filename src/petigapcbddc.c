#include "petiga.h"

static PetscErrorCode OrthonormalizeVecs_Private(PetscInt n, Vec vecs[])
{
  PetscInt       i,j;
  PetscScalar    *alphas;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc1(n,&alphas);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    ierr = VecMDot(vecs[i],i,vecs,alphas);CHKERRQ(ierr);
    for (j=0; j<i; j++) alphas[j] *= -1.;
    ierr = VecMAXPY(vecs[i],i,alphas,vecs);CHKERRQ(ierr);
    ierr = VecNormalize(vecs[i],NULL);CHKERRQ(ierr);
  }
  ierr = PetscFree(alphas);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
  if (i<L[0]) {r = i - L[0];} if (i>R[0]) {r = i - R[0];}
  if (j<L[1]) {g = j - L[1];} if (j>R[1]) {g = j - R[1];}
  if (k<L[2]) {b = k - L[2];} if (k>R[2]) {b = k - R[2];}
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
  nvtx *= bs; nadj *= bs*bs; /* adjust for block size */
  ierr = PetscMalloc1((size_t)(nvtx+1),&xadj);CHKERRQ(ierr);
  ierr = PetscMalloc1((size_t)nadj,&adjy);CHKERRQ(ierr);

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
          PetscInt c2;

          xadj[pos] = xadj[pos-1];
          for (v=0; v<nv; v++)
            for (c2=0; c2<bs; c2++)
              adjy[xadj[pos]++] = c2 + bs*vertices[v];
          pos++;
        }
      }

  *_nvtx = nvtx;
  *_xadj = xadj;
  *_adjy = adjy;

  PetscFunctionReturn(0);
}

static
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

  ierr = PetscMalloc1((size_t)ndirichlet,&idirichlet);CHKERRQ(ierr);
  for (pos=0,i=0; i<m; i++) if (PetscBTLookup(Dmask,i)) idirichlet[pos++] = i;
  ierr = PetscBTDestroy(&Dmask);CHKERRQ(ierr);
  *_ndirichlet = ndirichlet;
  *_idirichlet = idirichlet;

  ierr = PetscMalloc1((size_t)nneumann,&ineumann);CHKERRQ(ierr);
  for (pos=0,i=0; i<m; i++) if (PetscBTLookup(Nmask,i)) ineumann[pos++] = i;
  ierr = PetscBTDestroy(&Nmask);CHKERRQ(ierr);
  *_nneumann = nneumann;
  *_ineumann = ineumann;

  PetscFunctionReturn(0);
}

PetscErrorCode IGAPreparePCBDDC(IGA iga,PC pc)
{
  Mat                    mat;
  ISLocalToGlobalMapping l2g;
  const char             *prefix;
  PetscInt               num;
  PetscBool              boundary[2] = {PETSC_TRUE,PETSC_TRUE};
  PetscBool              minimal = PETSC_FALSE;
  PetscBool              primal = PETSC_TRUE;
  PetscBool              graph = PETSC_FALSE;
  PetscBool              isbddc;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(pc,PC_CLASSID,2);
  IGACheckSetUpStage2(iga,1);

  ierr = PetscObjectTypeCompare((PetscObject)pc,PCBDDC,&isbddc);CHKERRQ(ierr);
  if (!isbddc) PetscFunctionReturn(0);

  {
    Mat A,B;
    PetscBool useAmat = PETSC_FALSE;
    ierr = PCGetOperators(pc,&A,&B);CHKERRQ(ierr);
    ierr = PCGetUseAmat(pc,&useAmat);CHKERRQ(ierr);
    mat = useAmat ? A : B;
  }

  ierr = IGAGetOptionsPrefix(iga,&prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(((PetscObject)pc)->options,prefix,"-iga_set_bddc_graph",&graph,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(((PetscObject)pc)->options,prefix,"-iga_set_bddc_primal",&primal,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBoolArray(((PetscObject)pc)->options,prefix,"-iga_set_bddc_boundary",boundary,(num=2,&num),NULL);CHKERRQ(ierr);
  if (num == 1) boundary[1] = boundary[0];
  ierr = PetscOptionsGetBool(((PetscObject)pc)->options,prefix,"-iga_set_bddc_minimal",&minimal,NULL);CHKERRQ(ierr);

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
    ierr = PCBDDCSetLocalAdjacencyGraph(pc,nvtx,xadj,adjy,PETSC_OWN_POINTER);CHKERRQ(ierr);
  }

  if (primal) {
    PetscInt  i,j,k;
    PetscInt  dim,dof;
    PetscBool wrap[3] = {PETSC_FALSE,PETSC_FALSE,PETSC_FALSE};
    PetscInt  *rank = iga->proc_ranks;
    PetscInt  *shape = iga->node_gwidth;
    PetscInt  np=0,ip[8];
    MPI_Comm  comm;
    IS        isp;
    ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
    ierr = IGAGetDof(iga,&dof);CHKERRQ(ierr);
    for (i=0; i<dim; i++) wrap[i] = iga->axis[i]->periodic;
#define forall1(a) for ((a)=0; (a)<2; (a)++)
#define forall2(a,b) forall1(a) forall1(b)
#define forall3(a,b,c) forall2(a,b) forall1(c)
    {
      PetscInt  vertex[2][2][2],*index = &vertex[0][0][0];
      PetscInt  corner[3] = {0,0,0};
      for (i=0; i<dim; i++) {
        PetscInt lend = iga->node_lstart[i] + iga->node_lwidth[i];
        PetscInt gend = iga->node_gstart[i] + iga->node_gwidth[i];
        corner[i] = iga->node_gwidth[i] - (gend-lend)/2 - 1;
      }
      forall3(i,j,k) {
        PetscInt a = i ? corner[0] : 0;
        PetscInt b = j ? corner[1] : 0;
        PetscInt c = k ? corner[2] : 0;
        vertex[i][j][k] = Index(shape,a,b,c);
      }
      if (dim < 3) forall2(i,j) vertex[i][j][1] = -1;
      if (dim < 2) forall1(i)   vertex[i][1][0] = -1;
      if (rank[0] > 0 || wrap[0]) forall2(j,k) vertex[0][j][k] = -1;
      if (rank[1] > 0 || wrap[1]) forall2(i,k) vertex[i][0][k] = -1;
      if (rank[2] > 0 || wrap[2]) forall2(i,j) vertex[i][j][0] = -1;
      for (i=0; i<8; i++) if (index[i] >= 0) ip[np++] = index[i];
    }
#undef forall1
#undef forall2
#undef forall3
    ierr = PetscSortInt(np,ip);CHKERRQ(ierr);
    ierr = PetscObjectGetComm((PetscObject)pc,&comm);CHKERRQ(ierr);
    ierr = ISCreateBlock(comm,dof,np,ip,PETSC_COPY_VALUES,&isp);CHKERRQ(ierr);
    ierr = PCBDDCSetPrimalVerticesLocalIS(pc,isp);CHKERRQ(ierr);
    ierr = ISDestroy(&isp);CHKERRQ(ierr);
  }

  ierr = MatGetLocalToGlobalMapping(mat,&l2g,NULL);CHKERRQ(ierr);
  if (!l2g) l2g = iga->map->mapping;
  if (minimal && l2g) {
    MatNullSpace      nnsp;
    Vec               *nnsp_v = NULL,*v, mask, fat;
    PetscScalar       *vals,*ma;
    const PetscScalar *fa;
    PetscInt          nl,dim,nnsp_size,n,i,s,ni,bs,*idxs,nv,*minimalv;
    PetscInt          mid[3] = {0,0,0}, fix[3] = {0,0,0};
    PetscInt          width[3][2] = {{0,1},{0,1},{0,1}};
    PetscBool         nnsp_has_cnst = PETSC_TRUE,hasv;

    nnsp_size = 0;
    ierr = MatGetNearNullSpace(mat,&nnsp);CHKERRQ(ierr);
    if (nnsp) {
      ierr = MatNullSpaceGetVecs(nnsp,&nnsp_has_cnst,&nnsp_size,(const Vec**)&nnsp_v);CHKERRQ(ierr);
    }
    ierr = PetscMalloc1(nnsp_size+1,&minimalv);CHKERRQ(ierr);
    for (i=0;i<nnsp_size;i++) minimalv[i] = -1;
    ierr = PetscOptionsGetIntArray(((PetscObject)pc)->options,prefix,"-iga_set_bddc_minimal_volume",minimalv,(nv = nnsp_size,&nv),&hasv);CHKERRQ(ierr);
    if (!hasv) nv = 0;
    s = nnsp_has_cnst ? 1 : 0;
    n = nnsp_size + s + nv;

    ierr = PetscMalloc1(n,&v);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetSize(l2g,&ni);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetBlockSize(l2g,&bs);CHKERRQ(ierr);
    ierr = PetscMalloc2(ni/bs,&idxs,ni,&vals);CHKERRQ(ierr);
    for (i=0;i<ni/bs;i++) {
      PetscInt j;

      idxs[i] = i;
      for (j=0;j<bs;j++) vals[bs*i+j] = 1.0;
    }

    for (i=0;i<n;i++) {
      ierr = IGACreateVec(iga,&v[i]);CHKERRQ(ierr);
    }

    ierr = IGACreateVec(iga,&fat);CHKERRQ(ierr);
    ierr = VecSetValuesBlockedLocal(fat,ni/bs,idxs,vals,ADD_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(fat);CHKERRQ(ierr);

    ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
    for (i=0;i<dim;i++) {
      PetscInt lend = iga->node_lstart[i] + iga->node_lwidth[i];
      PetscInt gend = iga->node_gstart[i] + iga->node_gwidth[i];
      if (lend == gend-1 || lend == gend) fix[i] = 1; /* fix for lowest order */
      mid[i]        = iga->node_gwidth[i] - (gend-lend)/2 - 1;
      width[i][0]   = iga->node_lstart[i] - iga->node_gstart[i];
      width[i][1]   = width[i][0] + iga->node_lwidth[i];
    }
    for (i=0;i<ni;i++) vals[i] = 0.0;
    if (0 < mid[0] && mid[0] < iga->node_gwidth[0]-1 + fix[0]) {
      PetscInt j,k;
      for (k = width[2][0]; k < width[2][1]; k++) {
        for (j = width[1][0]; j < width[1][1]; j++) {
          PetscInt b, ii = k*iga->node_gwidth[0]*iga->node_gwidth[1] + j*iga->node_gwidth[0] + mid[0];
          for (b=0;b<bs;b++) vals[bs*ii+b] = 1.0;
        }
      }
    }
    if (0 < mid[1] && mid[1] < iga->node_gwidth[1]-1 + fix[0]) {
      PetscInt j,k;
      for (k = width[2][0]; k < width[2][1]; k++) {
        for (j = width[0][0]; j < width[0][1]; j++) {
          PetscInt b, ii = k*iga->node_gwidth[0]*iga->node_gwidth[1] + mid[1]*iga->node_gwidth[0] + j;
          for (b=0;b<bs;b++) vals[bs*ii+b] = 1.0;
        }
      }
    }
    if (0 < mid[2] && mid[2] < iga->node_gwidth[2]-1 + fix[0]) {
      PetscInt j,k;
      for (k = width[1][0]; k < width[1][1]; k++) {
        for (j = width[0][0]; j < width[0][1]; j++) {
          PetscInt b, ii = mid[2]*iga->node_gwidth[0]*iga->node_gwidth[1] + k*iga->node_gwidth[0] + j;
          for (b=0;b<bs;b++) vals[bs*ii+b] = 1.0;
        }
      }
    }
    ierr = IGACreateVec(iga,&mask);CHKERRQ(ierr);
    ierr = VecSetValuesBlockedLocal(mask,ni/bs,idxs,vals,ADD_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(mask);CHKERRQ(ierr);

    ierr = VecAssemblyEnd(fat);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)fat,"FATINT");CHKERRQ(ierr);
    ierr = VecViewFromOptions(fat,NULL,"-view_fat");CHKERRQ(ierr);
    ierr = VecAssemblyEnd(mask);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)mask,"INITMASK");CHKERRQ(ierr);
    ierr = VecViewFromOptions(mask,NULL,"-view_init_mask");CHKERRQ(ierr);
    ierr = VecGetArrayRead(fat,&fa);CHKERRQ(ierr);
    ierr = VecGetArray(mask,&ma);CHKERRQ(ierr);
    ierr = VecGetLocalSize(mask,&nl);CHKERRQ(ierr);
    for (i=0;i<nl;i++) {
      /* m = (2.^m == f) */
      PetscReal t = PetscPowReal(2.0,PetscRealPart(ma[i]));
      ma[i] = (PetscScalar)(PetscAbsReal(t-PetscRealPart(fa[i])) < PETSC_SMALL ? 1.0 : 0.0);
    }
    ierr = VecRestoreArrayRead(fat,&fa);CHKERRQ(ierr);
    ierr = VecRestoreArray(mask,&ma);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)mask,"MASK");CHKERRQ(ierr);
    ierr = VecViewFromOptions(mask,NULL,"-view_mask");CHKERRQ(ierr);
    if (s) {
      ierr = VecCopy(mask,v[0]);CHKERRQ(ierr);
    }
    for (i=0;i<nnsp_size;i++) {
      ierr = VecPointwiseMult(v[i+s],nnsp_v[i],mask);CHKERRQ(ierr);
    }
    for (i=0;i<nv;i++) {
      if (minimalv[i] < 0 || minimalv[i] >= nnsp_size) SETERRQ1(PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"Invalid volume term %D",minimalv[i]);
      ierr = VecCopy(nnsp_v[minimalv[i]],v[i+s+nnsp_size]);CHKERRQ(ierr);
    }
    ierr = OrthonormalizeVecs_Private(n,v);CHKERRQ(ierr);
    ierr = MatNullSpaceCreate(PetscObjectComm((PetscObject)mat),PETSC_FALSE,n,v,&nnsp);CHKERRQ(ierr);
    ierr = MatSetNearNullSpace(mat,nnsp);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nnsp);CHKERRQ(ierr);
    for (i=0;i<n;i++) {
      ierr = VecDestroy(&v[i]);CHKERRQ(ierr);
    }
    ierr = VecDestroy(&mask);CHKERRQ(ierr);
    ierr = VecDestroy(&fat);CHKERRQ(ierr);
    ierr = PetscFree(v);CHKERRQ(ierr);
    ierr = PetscFree2(idxs,vals);CHKERRQ(ierr);
    ierr = PetscFree(minimalv);CHKERRQ(ierr);
  }

  if (boundary[0] || boundary[1]) {
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
    ierr = PetscObjectGetComm((PetscObject)pc,&comm);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm,nd,id,PETSC_OWN_POINTER,&isd);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm,nn,in,PETSC_OWN_POINTER,&isn);CHKERRQ(ierr);
    if (boundary[0]) {
      ierr = PCBDDCSetDirichletBoundariesLocal(pc,isd);CHKERRQ(ierr);
    }
    if (boundary[1]) {
      ierr = PCBDDCSetNeumannBoundariesLocal(pc,isn);CHKERRQ(ierr);
    }
    ierr = ISDestroy(&isd);CHKERRQ(ierr);
    ierr = ISDestroy(&isn);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}
