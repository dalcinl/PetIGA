#include "petiga.h"
#include "petigagrid.h"

#if PETSC_VERSION_LT(3,9,0)
#define MatGetOperation MatShellGetOperation
#define MatSetOperation MatShellSetOperation
#endif

#if PETSC_VERSION_LT(3,9,0)
#define MatSeqSELLSetPreallocation(A,dnz,dnnz) 0
#define MatMPISELLSetPreallocation(A,dnz,dnnz,onz,onnz) 0
#endif

#if PETSC_VERSION_LT(3,8,0)
#define MatCreateSubMatrix MatGetSubMatrix
#define MATOP_CREATE_VECS MATOP_GET_VECS
#endif

static PetscErrorCode MatView_MPI_IGA(Mat,PetscViewer);
static PetscErrorCode MatLoad_MPI_IGA(Mat,PetscViewer);
static PetscErrorCode MatCreateVecs_IGA(Mat,Vec*,Vec*);

static PetscErrorCode MatView_MPI_IGA(Mat A,PetscViewer viewer)
{
  PetscViewerFormat format;
  MPI_Comm          comm;
  IGA               iga;
  Mat               Anatural;
  IS                is;
  PetscInt          bs,rstart,rend;
  const char        *prefix;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO) PetscFunctionReturn(0);
  if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) PetscFunctionReturn(0);

  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)A,"IGA",(PetscObject*)&iga);CHKERRQ(ierr);
  if (!iga) SETERRQ(comm,PETSC_ERR_ARG_WRONG,"Matrix not generated from IGA");
  PetscValidHeaderSpecific(iga,IGA_CLASSID,0);

  /* Map natural ordering to PETSc ordering and create IS */
  ierr = IGAGetDof(iga,&bs);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  ierr = ISCreateStride(comm,(rend-rstart)/bs,rstart/bs,1,&is);CHKERRQ(ierr);
  ierr = AOApplicationToPetscIS(iga->ao,is);CHKERRQ(ierr);
  if (bs > 1) {
    IS isb;
    PetscInt n;
    const PetscInt *idx;
    ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
    ierr = ISGetIndices(is,&idx);CHKERRQ(ierr);
    ierr = ISCreateBlock(comm,bs,n,idx,PETSC_COPY_VALUES,&isb);CHKERRQ(ierr);
    ierr = ISRestoreIndices(is,&idx);CHKERRQ(ierr);
    ierr = ISDestroy(&is);CHKERRQ(ierr);
    is = isb;
  }

  /* Do permutation and view matrix */
  ierr = MatCreateSubMatrix(A,is,is,MAT_INITIAL_MATRIX,&Anatural);CHKERRQ(ierr);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)A,&prefix);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)Anatural,prefix);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)Anatural,((PetscObject)A)->name);CHKERRQ(ierr);
  ierr = MatView(Anatural,viewer);CHKERRQ(ierr);

  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = MatDestroy(&Anatural);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode MatLoad_MPI_IGA(Mat A,PetscViewer viewer)
{
  MPI_Comm       comm;
  IGA            iga;
  MatType        mtype;
  PetscInt       rbs,cbs,m,n,M,N;
  Mat            Anatural,Apetsc;
  PetscInt       bs,rstart,rend;
  IS             is;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);

  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)A,"IGA",(PetscObject*)&iga);CHKERRQ(ierr);
  if (!iga) SETERRQ(comm,PETSC_ERR_ARG_WRONG,"Matrix not generated from IGA");
  PetscValidHeaderSpecific(iga,IGA_CLASSID,0);

  /* Create and load the matrix in natural ordering */
  ierr = MatGetType(A,&mtype);CHKERRQ(ierr);
  ierr = MatGetBlockSizes(A,&rbs,&cbs);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
  ierr = MatCreate(comm,&Anatural);CHKERRQ(ierr);
  ierr = MatSetType(Anatural,mtype);CHKERRQ(ierr);
  ierr = MatSetSizes(Anatural,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetBlockSizes(Anatural,rbs,cbs);CHKERRQ(ierr);
  ierr = MatLoad(Anatural,viewer);CHKERRQ(ierr);

  /* Map PETSc ordering to natural ordering and create IS */
  ierr = IGAGetDof(iga,&bs);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  ierr = ISCreateStride(comm,(rend-rstart)/bs,rstart/bs,1,&is);CHKERRQ(ierr);
  ierr = AOPetscToApplicationIS(iga->ao,is);CHKERRQ(ierr);
  if (bs > 1) {
    IS isb;
    PetscInt nidx;
    const PetscInt *idx;
    ierr = ISGetLocalSize(is,&nidx);CHKERRQ(ierr);
    ierr = ISGetIndices(is,&idx);CHKERRQ(ierr);
    ierr = ISCreateBlock(comm,bs,nidx,idx,PETSC_COPY_VALUES,&isb);CHKERRQ(ierr);
    ierr = ISRestoreIndices(is,&idx);CHKERRQ(ierr);
    ierr = ISDestroy(&is);CHKERRQ(ierr);
    is = isb;
  }

  /* Do permutation and copy values */
  ierr = MatCreateSubMatrix(Anatural,is,is,MAT_INITIAL_MATRIX,&Apetsc);CHKERRQ(ierr);
  ierr = MatCopy(Apetsc,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);

  ierr = MatDestroy(&Anatural);CHKERRQ(ierr);
  ierr = MatDestroy(&Apetsc);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode MatCreateVecs_IGA(Mat A,Vec *right,Vec *left)
{
  IGA            iga;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)A,"IGA",(PetscObject*)&iga);CHKERRQ(ierr);
  if (!iga) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Matrix not generated from IGA");
  PetscValidHeaderSpecific(iga,IGA_CLASSID,0);

  if (right) {ierr = IGACreateVec(iga,right);CHKERRQ(ierr);}
  if (left)  {ierr = IGACreateVec(iga,left );CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDuplicate_IGA(Mat A,MatDuplicateOption op,Mat *B)
{
  IGA            iga;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(B,3);
  ierr = PetscObjectQuery((PetscObject)A,"IGA",(PetscObject*)&iga);CHKERRQ(ierr);
  if (!iga) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Matrix not generated from IGA");
  PetscValidHeaderSpecific(iga,IGA_CLASSID,0);

  { /* MatDuplicate */
    PetscErrorCode (*matduplicate)(Mat,MatDuplicateOption,Mat*);
    ierr = PetscObjectQueryFunction((PetscObject)A,"__IGA_MatDuplicate",&matduplicate);CHKERRQ(ierr);
    ierr = matduplicate(A,op,B);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)*B,"IGA",(PetscObject)iga);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)*B,"__IGA_MatDuplicate",matduplicate);CHKERRQ(ierr);
    ierr = MatSetOperation(*B,MATOP_DUPLICATE,(PetscVoidFunction)MatDuplicate_IGA);CHKERRQ(ierr);
  }
  {  /* MatView & MatLoad */
    PetscErrorCode (*matview)(Mat,PetscViewer);
    PetscErrorCode (*matload)(Mat,PetscViewer);
    ierr = MatGetOperation(A,MATOP_VIEW,(PetscVoidFunction*)&matview);CHKERRQ(ierr);
    ierr = MatGetOperation(A,MATOP_LOAD,(PetscVoidFunction*)&matload);CHKERRQ(ierr);
    if (matview == MatView_MPI_IGA) {ierr = MatSetOperation(*B,MATOP_VIEW,(PetscVoidFunction)matview);CHKERRQ(ierr);}
    if (matload == MatLoad_MPI_IGA) {ierr = MatSetOperation(*B,MATOP_LOAD,(PetscVoidFunction)matload);CHKERRQ(ierr);}
  }
  {  /* MatCreateVecs */
    ierr = MatSetOperation(*B,MATOP_CREATE_VECS,(PetscVoidFunction)MatCreateVecs_IGA);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
extern PetscInt IGA_NextKnot(PetscInt m,const PetscReal U[],PetscInt k,PetscInt direction);
EXTERN_C_END

PETSC_STATIC_INLINE
void Stencil(IGA iga,PetscInt dir,PetscInt i,PetscInt *first,PetscInt *last)
{
  IGAAxis axis = iga->axis[dir];
  PetscInt  p  = axis->p;
  PetscInt  m  = axis->m;
  PetscInt  n  = m - p - 1;
  PetscReal *U = axis->U;

  if (PetscUnlikely(iga->collocation)) {
    PetscInt k = iga->basis[dir]->offset[i];
    *first = k; *last  = k + p; return;
  }

  { /* compute index of the leftmost overlapping basis */
    PetscInt k = i;
    k = IGA_NextKnot(m,U,k,+1);
    *first = k - p - 1;
  }

  { /* compute index of the rightmost overlapping basis */
    PetscInt k = i + p + 1;
    k = IGA_NextKnot(m,U,k,-1);
    *last = k;
  }

  if (!axis->periodic) {
    if (i <= p)   *first = 0;
    if (i >= n-p) *last  = n;
  } else if (i==0) {
    PetscInt k = n+1;
    PetscInt j = IGA_NextKnot(m,U,k,+1);
    PetscInt s = j-k, C = p-s, nnp = n-C;
    k = IGA_NextKnot(m,U,nnp,+1) - nnp;
    *first = k - p - 1;
  }
}

PETSC_STATIC_INLINE
PetscInt Index3D(const PetscInt start[3],const PetscInt shape[3],
                 PetscInt i,PetscInt j,PetscInt k)
{
  if (start) { i -= start[0]; j -= start[1]; k -= start[2]; }
  return i + j * shape[0] + k * shape[0] * shape[1];
}

PETSC_STATIC_INLINE
PetscInt ColumnIndices(IGA iga,const PetscInt start[3],const PetscInt shape[3],
                       PetscInt iA,PetscInt jA,PetscInt kA,PetscInt stencil[])
{
  PetscInt dim = PetscClipInterval(iga->dim,1,3);
  PetscInt first[3] = {0,0,0};
  PetscInt last [3] = {0,0,0};
  PetscInt count    = 0;
  { /* compute range of overlapping basis in each direction */
    PetscInt i,A[3]; A[0] = iA; A[1] = jA; A[2] = kA;
    for (i=0; i<dim; i++) {
      Stencil(iga,i,A[i],&first[i],&last[i]);
      first[i] = PetscMax(first[i],start[i]);
      last [i] = PetscMin(last [i],start[i]+shape[i]-1);
    }
  }
  { /* tensor-product the ranges of overlapping basis */
    PetscInt i,j,k;
    for (k=first[2]; k<=last[2]; k++)
    for (j=first[1]; j<=last[1]; j++)
    for (i=first[0]; i<=last[0]; i++)
    stencil[count++] = Index3D(start,shape,i,j,k);
  }
  return count;
}

PETSC_STATIC_INLINE
PetscErrorCode InferMatrixType(Mat A,PetscBool *aij,PetscBool *baij,PetscBool *sbaij,PetscBool *sell)
{
  void (*f)(void) = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *aij = *baij = *sbaij = *sell = PETSC_FALSE;
  if (!f) {ierr = PetscObjectQueryFunction((PetscObject)A,"MatMPIAIJSetPreallocation_C",&f);CHKERRQ(ierr);}
  if (!f) {ierr = PetscObjectQueryFunction((PetscObject)A,"MatSeqAIJSetPreallocation_C",&f);CHKERRQ(ierr);}
  if  (f) {*aij = PETSC_TRUE; goto done;};
  if (!f) {ierr = PetscObjectQueryFunction((PetscObject)A,"MatMPIBAIJSetPreallocation_C",&f);CHKERRQ(ierr);}
  if (!f) {ierr = PetscObjectQueryFunction((PetscObject)A,"MatSeqBAIJSetPreallocation_C",&f);CHKERRQ(ierr);}
  if  (f) {*baij = PETSC_TRUE; goto done;};
  if (!f) {ierr = PetscObjectQueryFunction((PetscObject)A,"MatMPISBAIJSetPreallocation_C",&f);CHKERRQ(ierr);}
  if (!f) {ierr = PetscObjectQueryFunction((PetscObject)A,"MatSeqSBAIJSetPreallocation_C",&f);CHKERRQ(ierr);}
  if  (f) {*sbaij = PETSC_TRUE; goto done;};
  if (!f) {ierr = PetscObjectQueryFunction((PetscObject)A,"MatMPISELLSetPreallocation_C",&f);CHKERRQ(ierr);}
  if (!f) {ierr = PetscObjectQueryFunction((PetscObject)A,"MatSeqSELLSetPreallocation_C",&f);CHKERRQ(ierr);}
  if  (f) {*sell = PETSC_TRUE; goto done;};
  done:
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE
PetscErrorCode L2GApplyBlock(ISLocalToGlobalMapping ltog,PetscInt *row,PetscInt *cnt,PetscInt col[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = ISLocalToGlobalMappingApplyBlock(ltog,1,row,row);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApplyBlock(ltog,*cnt,col,col);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE
PetscErrorCode UnblockIndices(PetscInt bs,PetscInt row,PetscInt count,const PetscInt indices[],PetscInt ubrows[],PetscInt ubcols[])
{
  PetscInt n,c;
  PetscFunctionBegin;
  for (c=0; c<bs; c++)
    ubrows[c] = c + row*bs;
  for (n=0; n<count; n++)
    for (c=0; c<bs; c++)
      ubcols[c+n*bs] = c + indices[n]*bs;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE
PetscErrorCode FilterLowerTriangular(PetscInt row,PetscInt *cnt,PetscInt col[])
{
  PetscInt i,n;
  PetscFunctionBegin;
  for (i=0,n=0; i<*cnt; i++)
    if (col[i] >= row)
      col[n++] = col[i];
  *cnt = n;
  PetscFunctionReturn(0);
}

/*@
   IGACreateMat - Creates a matrix with the correct parallel layout
   required for computing a matrix using the discretization
   information provided in the IGA.

   Collective on IGA

   Input Parameter:
.  iga - the IGA context

   Output Parameter:
.  mat - the matrix with properly allocated nonzero structure

   Level: normal

.keywords: IGA, create, matrix
@*/
PetscErrorCode IGACreateMat(IGA iga,Mat *mat)
{
  MPI_Comm       comm;
  PetscMPIInt    size;
  PetscBool      is,aij,baij,sbaij,sell;
  PetscInt       i,j,k,dim;
  PetscInt       *lstart,*lwidth;
  PetscInt       gstart[3] = {0,0,0};
  PetscInt       gwidth[3] = {1,1,1};
  PetscInt       maxnnz;
  PetscInt       bs,n,N;
  PetscLayout    rmap,cmap;
  LGMap          ltog = NULL;
  Mat            A;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(mat,2);
  IGACheckSetUpStage2(iga,1);

  ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
  ierr = IGAGetDof(iga,&bs);CHKERRQ(ierr);

  rmap = cmap = iga->map;
  ierr = MatCreate(comm,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,rmap->n,cmap->n,rmap->N,cmap->N);CHKERRQ(ierr);
  ierr = MatSetBlockSizes(A,rmap->bs,cmap->bs);CHKERRQ(ierr);
  ierr = MatSetType(A,iga->mattype);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)A,"IGA",(PetscObject)iga);CHKERRQ(ierr);
  *mat = A;

  { /* Check for MATIS matrix subtype */
    PetscVoidFunction f = NULL;
    ierr = PetscObjectQueryFunction((PetscObject)A,"MatISGetLocalMat_C",&f);CHKERRQ(ierr);
    is = f ? PETSC_TRUE: PETSC_FALSE;
  }

  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (!is && size > 1) { /* Change MatView/MatLoad to handle matrix in natural ordering */
    ierr = MatSetOperation(A,MATOP_VIEW,(PetscVoidFunction)MatView_MPI_IGA);CHKERRQ(ierr);
    ierr = MatSetOperation(A,MATOP_LOAD,(PetscVoidFunction)MatLoad_MPI_IGA);CHKERRQ(ierr);
  }
  { /* Change MatCreateVecs to propagate composed objects */
    ierr = MatSetOperation(A,MATOP_CREATE_VECS,(PetscVoidFunction)MatCreateVecs_IGA);CHKERRQ(ierr);
  }
  { /* Change MatDuplicate to propagate composed objects and method overrides */
    PetscErrorCode (*matduplicate)(Mat,MatDuplicateOption,Mat*);
    ierr = MatGetOperation(A,MATOP_DUPLICATE,(PetscVoidFunction*)&matduplicate);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)A,"__IGA_MatDuplicate",matduplicate);CHKERRQ(ierr);
    ierr = MatSetOperation(A,MATOP_DUPLICATE,(PetscVoidFunction)MatDuplicate_IGA);CHKERRQ(ierr);
  }

  ierr = MatSetLocalToGlobalMapping(A,rmap->mapping,cmap->mapping);CHKERRQ(ierr);
  if (is) {
    const MatType mtype = (bs > 1) ? MATBAIJ : MATAIJ;
    ierr = MatISGetLocalMat(A,&A);CHKERRQ(ierr);
    ierr = MatSetType(A,mtype);CHKERRQ(ierr);
    ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  }

  ierr = InferMatrixType(A,&aij,&baij,&sbaij,&sell);CHKERRQ(ierr);

  if (!is) {
    lstart = iga->node_lstart;
    lwidth = iga->node_lwidth;
    for (i=0; i<dim; i++) {
      PetscInt gfirst, first = lstart[i];
      PetscInt glast,  last  = lstart[i] + lwidth[i] - 1;
      Stencil(iga,i,first,&gstart[i],&glast);
      Stencil(iga,i,last,&gfirst,&glast);
      gwidth[i] = glast + 1 - gstart[i];
    }
    if (aij || baij || sbaij || sell) {
      IGA_Grid  grid;
      PetscInt *sizes = iga->node_sizes;
      ierr = IGA_Grid_Create(comm,&grid);CHKERRQ(ierr);
      ierr = IGA_Grid_Init(grid,iga->dim,1,sizes,lstart,lwidth,gstart,gwidth);CHKERRQ(ierr);
      ierr = IGA_Grid_SetAO(grid,iga->ao);CHKERRQ(ierr);
      ierr = IGA_Grid_GetLGMap(grid,&ltog);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)ltog);CHKERRQ(ierr);
      ierr = IGA_Grid_Destroy(&grid);CHKERRQ(ierr);
    }
  } else {
    lstart = iga->node_gstart;
    lwidth = iga->node_gwidth;
    for (i=0; i<dim; i++) {
      gstart[i] = lstart[i];
      gwidth[i] = lwidth[i];
    }
  }

  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&n,NULL);CHKERRQ(ierr);
  ierr = MatGetSize(A,NULL,&N);CHKERRQ(ierr);
  n /= bs; N /= bs;


  maxnnz = 1;
  for (i=0; i<dim; i++)
    maxnnz *= (2*iga->axis[i]->p + 1); /* XXX do better ? */

  if (aij || baij || sbaij || sell) {
    PetscInt nbs = (baij||sbaij) ? n : n*bs;
    PetscInt Nbs = (baij||sbaij) ? N : N*bs;
    PetscInt *dnz = NULL, *onz = NULL;
    ierr = MatPreallocateInitialize(comm,nbs,nbs,dnz,onz);CHKERRQ(ierr);
    {
      PetscInt nnz = maxnnz,*indices=NULL,*ubrows=NULL,*ubcols=NULL;
      ierr = PetscMalloc1((size_t)nnz,&indices);CHKERRQ(ierr);
      ierr = PetscMalloc2((size_t)bs,&ubrows,(size_t)(nnz*bs),&ubcols);CHKERRQ(ierr);
      for (k=lstart[2]; k<lstart[2]+lwidth[2]; k++)
        for (j=lstart[1]; j<lstart[1]+lwidth[1]; j++)
          for (i=lstart[0]; i<lstart[0]+lwidth[0]; i++)
            { /* */
              PetscInt r,row = Index3D(gstart,gwidth,i,j,k);
              PetscInt count = ColumnIndices(iga,gstart,gwidth,i,j,k,indices);
              if (ltog) {ierr = L2GApplyBlock(ltog,&row,&count,indices);CHKERRQ(ierr);}
              if (aij || sell) {
                if (bs == 1) {
                  ierr = MatPreallocateSet(row,count,indices,dnz,onz);CHKERRQ(ierr);
                } else {
                  ierr = UnblockIndices(bs,row,count,indices,ubrows,ubcols);CHKERRQ(ierr);
                  for (r=0; r<bs; r++) {
                    ierr = MatPreallocateSet(ubrows[r],count*bs,ubcols,dnz,onz);CHKERRQ(ierr);
                  }
                }
              } else if (baij) {
                ierr = MatPreallocateSet(row,count,indices,dnz,onz);CHKERRQ(ierr);
              } else if (sbaij) {
                ierr = FilterLowerTriangular(row,&count,indices);CHKERRQ(ierr);
                ierr = MatPreallocateSymmetricSetBlock(row,count,indices,dnz,onz);CHKERRQ(ierr);
              }
            } /* */
      ierr = PetscFree2(ubrows,ubcols);CHKERRQ(ierr);
      ierr = PetscFree(indices);CHKERRQ(ierr);
      if (N < maxnnz) {
        PetscInt dmaxnz = nbs;
        PetscInt omaxnz = Nbs - nbs;
        for (i=0; i<nbs; i++) {
          dnz[i] = PetscMin(dnz[i],dmaxnz);
          onz[i] = PetscMin(onz[i],omaxnz);
        }
      }
      if (aij) {
        ierr = MatSeqAIJSetPreallocation(A,0,dnz);CHKERRQ(ierr);
        ierr = MatMPIAIJSetPreallocation(A,0,dnz,0,onz);CHKERRQ(ierr);
      } else if (baij) {
        ierr = MatSeqBAIJSetPreallocation(A,bs,0,dnz);CHKERRQ(ierr);
        ierr = MatMPIBAIJSetPreallocation(A,bs,0,dnz,0,onz);CHKERRQ(ierr);
      } else if (sbaij) {
        ierr = MatSeqSBAIJSetPreallocation(A,bs,0,dnz);CHKERRQ(ierr);
        ierr = MatMPISBAIJSetPreallocation(A,bs,0,dnz,0,onz);CHKERRQ(ierr);
      } else if (sell) {
        ierr = MatSeqSELLSetPreallocation(A,0,dnz);CHKERRQ(ierr);
        ierr = MatMPISELLSetPreallocation(A,0,dnz,0,onz);CHKERRQ(ierr);
      }
    }
    ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  } else {
    ierr = MatSetUp(A);CHKERRQ(ierr);
  }

  if (aij || baij || sbaij || sell) {
    PetscInt nnz = maxnnz,*indices=NULL,*ubrows=NULL,*ubcols=NULL;PetscScalar *values=NULL;
    ierr = PetscMalloc2((size_t)bs,&ubrows,(size_t)(nnz*bs),&ubcols);CHKERRQ(ierr);
    ierr = PetscMalloc2((size_t)nnz,&indices,(size_t)(nnz*bs*nnz*bs),&values);CHKERRQ(ierr);
    ierr = PetscMemzero(values,(size_t)(nnz*bs*nnz*bs)*sizeof(PetscScalar));CHKERRQ(ierr);
    for (k=lstart[2]; k<lstart[2]+lwidth[2]; k++)
      for (j=lstart[1]; j<lstart[1]+lwidth[1]; j++)
        for (i=lstart[0]; i<lstart[0]+lwidth[0]; i++)
          { /* */
            PetscInt row   = Index3D(gstart,gwidth,i,j,k);
            PetscInt count = ColumnIndices(iga,gstart,gwidth,i,j,k,indices);
            if (ltog) {ierr = L2GApplyBlock(ltog,&row,&count,indices);CHKERRQ(ierr);}
            if (aij || sell) {
              if (bs == 1) {
                ierr = MatSetValues(A,1,&row,count,indices,values,INSERT_VALUES);CHKERRQ(ierr);
              } else {
                ierr = UnblockIndices(bs,row,count,indices,ubrows,ubcols);CHKERRQ(ierr);
                ierr = MatSetValues(A,bs,ubrows,count*bs,ubcols,values,INSERT_VALUES);CHKERRQ(ierr);
              }
            } else if (baij) {
              ierr = MatSetValuesBlocked(A,1,&row,count,indices,values,INSERT_VALUES);CHKERRQ(ierr);
            } else if (sbaij) {
              ierr = FilterLowerTriangular(row,&count,indices);CHKERRQ(ierr);
              ierr = MatSetValuesBlocked(A,1,&row,count,indices,values,INSERT_VALUES);CHKERRQ(ierr);
            }
          }
    ierr = PetscFree2(ubrows,ubcols);CHKERRQ(ierr);
    ierr = PetscFree2(indices,values);CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
    /*ierr = MatSetOption(A,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE);CHKERRQ(ierr);*/
    /*ierr = MatSetOption(A,MAT_STRUCTURALLY_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);*/
  }

  ierr = ISLocalToGlobalMappingDestroy(&ltog);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
