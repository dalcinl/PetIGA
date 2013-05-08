#include "petiga.h"
#include "petigagrid.h"
#if PETSC_VERSION_LE(3,2,0)
#include "private/matimpl.h"
#endif

#if PETSC_VERSION_LE(3,3,0)
#undef MatType
typedef const char* MatType;
#endif

PETSC_EXTERN PetscErrorCode MatHeaderReplace(Mat,Mat);
static       PetscErrorCode MatView_MPI_IGA(Mat,PetscViewer);
static       PetscErrorCode MatLoad_MPI_IGA(Mat,PetscViewer);

#undef  __FUNCT__
#define __FUNCT__ "MatView_MPI_IGA"
static PetscErrorCode MatView_MPI_IGA(Mat A,PetscViewer viewer)
{
  PetscViewerFormat format;
  MPI_Comm          comm;
  IGA               iga;
  Mat               Anatural;
  PetscInt          rstart,rend;
  IS                is;
  const char        *prefix;
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO       ) PetscFunctionReturn(0);
  if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) PetscFunctionReturn(0);

  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)A,"IGA",(PetscObject*)&iga);CHKERRQ(ierr);
  if (!iga) SETERRQ(comm,PETSC_ERR_ARG_WRONG,"Matrix not generated from a IGA");
  PetscValidHeaderSpecific(iga,IGA_CLASSID,0);

  /* Map natural ordering to PETSc ordering and create IS */
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  ierr = ISCreateStride(comm,rend-rstart,rstart,1,&is);CHKERRQ(ierr);
  ierr = AOApplicationToPetscIS(iga->ao,is);CHKERRQ(ierr);

  /* Do permutation and view matrix */
  ierr = MatGetSubMatrix(A,is,is,MAT_INITIAL_MATRIX,&Anatural);CHKERRQ(ierr);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)A,&prefix);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)Anatural,prefix);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)Anatural,((PetscObject)A)->name);CHKERRQ(ierr);
  ierr = MatView(Anatural,viewer);CHKERRQ(ierr);

  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = MatDestroy(&Anatural);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "MatLoad_MPI_IGA"
static PetscErrorCode MatLoad_MPI_IGA(Mat A,PetscViewer viewer)
{
  MPI_Comm       comm;
  IGA            iga;
  MatType        mtype;
  PetscInt       m,n,M,N;
  Mat            Anatural,Apetsc;
  PetscInt       rstart,rend;
  IS             is;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);

  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)A,"IGA",(PetscObject*)&iga);CHKERRQ(ierr);
  if (!iga) SETERRQ(comm,PETSC_ERR_ARG_WRONG,"Matrix not generated from a IGA");
  PetscValidHeaderSpecific(iga,IGA_CLASSID,0);

  /* Create and load the matrix in natural ordering */
  ierr = MatGetType(A,&mtype);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
  ierr = MatCreate(comm,&Anatural);CHKERRQ(ierr);
  ierr = MatSetType(Anatural,mtype);CHKERRQ(ierr);
  ierr = MatSetSizes(Anatural,m,n,M,N);CHKERRQ(ierr);
  ierr = MatLoad(Anatural,viewer);CHKERRQ(ierr);

  /* Map PETSc ordering to natural ordering and create IS */
  ierr = MatGetOwnershipRange(Anatural,&rstart,&rend);CHKERRQ(ierr);
  ierr = ISCreateStride(comm,rend-rstart,rstart,1,&is);CHKERRQ(ierr);
  ierr = AOPetscToApplicationIS(iga->ao,is);CHKERRQ(ierr);

  /* Do permutation and replace header */
  ierr = MatGetSubMatrix(Anatural,is,is,MAT_INITIAL_MATRIX,&Apetsc);CHKERRQ(ierr);
  ierr = MatHeaderReplace(A,Apetsc);CHKERRQ(ierr);

  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = MatDestroy(&Anatural);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


PETSC_STATIC_INLINE
void Stencil(IGA iga,PetscInt dir,PetscInt i,PetscInt *first,PetscInt *last)
{
  PetscBool periodic = iga->axis[dir]->periodic;
  PetscInt  p  = iga->axis[dir]->p;
  PetscInt  m  = iga->axis[dir]->m;
  PetscInt  n  = m - p - 1;
  PetscReal *U = iga->axis[dir]->U;
  PetscInt k;

  if (PetscUnlikely(iga->collocation)) {
    k = iga->basis[dir]->offset[i];
    *first = k; *last  = k + p; return;
  }

  /* compute index of the leftmost overlapping basis */
  k = i;
  while (U[k]==U[k+1]) k++; /* XXX Using "==" with floating point values ! */
  *first = k - p;

  /* compute index of the rightmost overlapping basis */
  k = i + p + 1;
  while (U[k]==U[k-1]) k--; /* XXX Using "==" with floating point values ! */
  *last = k - 1;

  if (!periodic) {
    if (i <= p  ) *first = 0;
    if (i >= n-p) *last  = n;
  } else if (i==0) {
    PetscInt s = 1;
    while (s < p && U[m-p]==U[m-p+s]) s++;
    k = n - p + s;
    while (U[k]==U[k+1]) k++;
    *first = k - s - n;
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
  PetscInt first[3] = {0,0,0};
  PetscInt last [3] = {0,0,0};
  PetscInt count    = 0;
  { /* compute range of overlapping basis in each direction */
    PetscInt i,A[3]; A[0] = iA; A[1] = jA; A[2] = kA;
    for (i=0; i<iga->dim; i++) {
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
#undef  __FUNCT__
#define __FUNCT__ "InferMatrixType"
PetscErrorCode InferMatrixType(Mat A,PetscBool *aij,PetscBool *baij,PetscBool *sbaij)
{
  void (*f)(void) = 0;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  *aij = *baij = *sbaij = PETSC_FALSE;
  if (!f) {ierr = PetscObjectQueryFunction((PetscObject)A,"MatMPIAIJSetPreallocation_C",&f);CHKERRQ(ierr);}
  if (!f) {ierr = PetscObjectQueryFunction((PetscObject)A,"MatSeqAIJSetPreallocation_C",&f);CHKERRQ(ierr);}
  if ( f) {*aij = PETSC_TRUE; goto done;};
  if (!f) {ierr = PetscObjectQueryFunction((PetscObject)A,"MatMPIBAIJSetPreallocation_C",&f);CHKERRQ(ierr);}
  if (!f) {ierr = PetscObjectQueryFunction((PetscObject)A,"MatSeqBAIJSetPreallocation_C",&f);CHKERRQ(ierr);}
  if ( f) {*baij = PETSC_TRUE; goto done;};
  if (!f) {ierr = PetscObjectQueryFunction((PetscObject)A,"MatMPISBAIJSetPreallocation_C",&f);CHKERRQ(ierr);}
  if (!f) {ierr = PetscObjectQueryFunction((PetscObject)A,"MatSeqSBAIJSetPreallocation_C",&f);CHKERRQ(ierr);}
  if ( f) {*sbaij = PETSC_TRUE; goto done;};
 done:
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE
#undef  __FUNCT__
#define __FUNCT__ "L2GApply"
PetscErrorCode L2GApply(ISLocalToGlobalMapping ltog,PetscInt *row,PetscInt *cnt,PetscInt col[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = ISLocalToGlobalMappingApply(ltog,1,row,row);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApply(ltog,*cnt,col,col);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE
#undef  __FUNCT__
#define __FUNCT__ "UnblockIndices"
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
#undef  __FUNCT__
#define __FUNCT__ "FilterLowerTriangular"
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

#undef  __FUNCT__
#define __FUNCT__ "IGACreateMat"
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
  PetscBool      is,aij,baij,sbaij;
  PetscInt       i,dim;
  PetscInt       *lstart,*lwidth;
  PetscInt       gstart[3] = {0,0,0};
  PetscInt       gwidth[3] = {1,1,1};
  PetscInt       maxnnz;
  PetscInt       n,N,bs;
  LGMap          ltogb = 0;
  Mat            A;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(mat,2);
  IGACheckSetUp(iga,1);

  ierr = PetscObjectGetComm((PetscObject)iga,&comm);CHKERRQ(ierr);
  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
  ierr = IGAGetDof(iga,&bs);CHKERRQ(ierr);

  n = N = 1;
  for(i=0; i<dim; i++) {
    n *= iga->node_lwidth[i];
    N *= iga->node_sizes[i];
  }
  ierr = MatCreate(comm,&A);CHKERRQ(ierr);
#if PETSC_VERSION_LE(3,2,0)
  ierr = MatSetType(A,iga->mattype);CHKERRQ(ierr);
  ierr = MatSetSizes(A,bs*n,bs*n,bs*N,bs*N);CHKERRQ(ierr);
#else
  ierr = MatSetSizes(A,bs*n,bs*n,bs*N,bs*N);CHKERRQ(ierr);
  ierr = MatSetBlockSize(A,bs);CHKERRQ(ierr);
  ierr = MatSetType(A,iga->mattype);CHKERRQ(ierr);
#endif
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)A,"IGA",(PetscObject)iga);CHKERRQ(ierr);
  *mat = A;

  { /* Check for MATIS matrix subtype */
    void (*f)(void) = 0;
    ierr = PetscObjectQueryFunction((PetscObject)A,"MatISGetLocalMat_C",&f);CHKERRQ(ierr);
    is = f ? PETSC_TRUE: PETSC_FALSE;
  }

  if (is) {ierr = MatSetUp(A);CHKERRQ(ierr);}
#if PETSC_VERSION_LE(3,2,0)
#else
  ierr = MatSetLocalToGlobalMapping(A,iga->lgmap,iga->lgmap);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMappingBlock(A,iga->lgmapb,iga->lgmapb);CHKERRQ(ierr);
#endif
  if (is) {
    const MatType mtype = (bs > 1) ? MATBAIJ : MATAIJ;
    ierr = MatISGetLocalMat(A,&A);CHKERRQ(ierr);
    ierr = MatSetType(A,mtype);CHKERRQ(ierr);
    ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  }

  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (!is && size > 1) { /* change viewer to display matrix in natural ordering */
    ierr = MatShellSetOperation(A,MATOP_VIEW,(void (*)(void))MatView_MPI_IGA);CHKERRQ(ierr);
    ierr = MatShellSetOperation(A,MATOP_LOAD,(void (*)(void))MatLoad_MPI_IGA);CHKERRQ(ierr);
  }

  ierr = InferMatrixType(A,&aij,&baij,&sbaij);CHKERRQ(ierr);

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
    if (aij || baij || sbaij) {
      IGA_Grid  grid;
      PetscInt *sizes = iga->node_sizes;
      ierr = IGA_Grid_Create(comm,&grid);CHKERRQ(ierr);
      ierr = IGA_Grid_Init(grid,iga->dim,bs,sizes,lstart,lwidth,gstart,gwidth);CHKERRQ(ierr);
      ierr = IGA_Grid_SetAOBlock(grid,iga->aob);CHKERRQ(ierr);
      ierr = IGA_Grid_GetLGMapBlock(grid,&ltogb);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)ltogb);CHKERRQ(ierr);
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
  ierr = MatGetLocalSize(A,&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatGetSize(A,PETSC_NULL,&N);CHKERRQ(ierr);
  n /= bs; N /= bs;

  maxnnz = 1;
  for(i=0; i<dim; i++)
    maxnnz *= (2*iga->axis[i]->p + 1); /* XXX do better ? */

  if (aij || baij || sbaij) {
    PetscInt nbs = (baij||sbaij) ? n : n*bs;
    PetscInt Nbs = (baij||sbaij) ? N : N*bs;
    PetscInt *dnz = 0, *onz = 0;
    ierr = MatPreallocateInitialize(comm,nbs,nbs,dnz,onz);CHKERRQ(ierr);
    {
      PetscInt i,j,k;
      PetscInt nnz = maxnnz,*indices=0,*ubrows=0,*ubcols=0;
      ierr = PetscMalloc1(nnz,PetscInt,&indices);CHKERRQ(ierr);
      ierr = PetscMalloc2(bs,PetscInt,&ubrows,nnz*bs,PetscInt,&ubcols);CHKERRQ(ierr);
      for (k=lstart[2]; k<lstart[2]+lwidth[2]; k++)
        for (j=lstart[1]; j<lstart[1]+lwidth[1]; j++)
          for (i=lstart[0]; i<lstart[0]+lwidth[0]; i++)
            { /* */
              PetscInt r,row = Index3D(gstart,gwidth,i,j,k);
              PetscInt count = ColumnIndices(iga,gstart,gwidth,i,j,k,indices);
              if (ltogb) {ierr = L2GApply(ltogb,&row,&count,indices);CHKERRQ(ierr);}
              if (aij) {
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
                ierr = MatPreallocateSymmetricSet(row,count,indices,dnz,onz);CHKERRQ(ierr);
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
      }
    }
    ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  } else {
    ierr = MatSetUp(A);CHKERRQ(ierr);
  }

#if PETSC_VERSION_LE(3,2,0)
  /* XXX This is a vile hack. Perhaps we should just check for      */
  /* SeqDense and MPIDense that are the only I care about right now */
  if ((*mat)->ops->setblocksize) {
    ierr = MatSetBlockSize(*mat,bs);CHKERRQ(ierr);
  } else {
    ierr = PetscLayoutSetBlockSize((*mat)->rmap,bs);CHKERRQ(ierr);
    ierr = PetscLayoutSetBlockSize((*mat)->cmap,bs);CHKERRQ(ierr);
  }
  if (!is) {
    ierr = MatSetLocalToGlobalMapping(*mat,iga->lgmap,iga->lgmap);CHKERRQ(ierr);
    ierr = MatSetLocalToGlobalMappingBlock(*mat,iga->lgmapb,iga->lgmapb);CHKERRQ(ierr);
  }
#endif

  if (aij || baij || sbaij) {
    PetscInt i,j,k;
    PetscInt nnz = maxnnz,*indices=0,*ubrows=0,*ubcols=0;PetscScalar *values=0;
    ierr = PetscMalloc2(nnz,PetscInt,&indices,nnz*bs*nnz*bs,PetscScalar,&values);CHKERRQ(ierr);
    ierr = PetscMalloc2(bs,PetscInt,&ubrows,nnz*bs,PetscInt,&ubcols);CHKERRQ(ierr);
    ierr = PetscMemzero(values,nnz*bs*nnz*bs*sizeof(PetscScalar));CHKERRQ(ierr);
    for (k=lstart[2]; k<lstart[2]+lwidth[2]; k++)
      for (j=lstart[1]; j<lstart[1]+lwidth[1]; j++)
        for (i=lstart[0]; i<lstart[0]+lwidth[0]; i++)
          { /* */
            PetscInt row   = Index3D(gstart,gwidth,i,j,k);
            PetscInt count = ColumnIndices(iga,gstart,gwidth,i,j,k,indices);
            if (ltogb) {ierr = L2GApply(ltogb,&row,&count,indices);CHKERRQ(ierr);}
            if (aij) {
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
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd  (A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
    /*ierr = MatSetOption(A,MAT_STRUCTURALLY_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);*/
    /*ierr = MatSetOption(A,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE);CHKERRQ(ierr);*/
  }

  ierr = ISLocalToGlobalMappingDestroy(&ltogb);CHKERRQ(ierr);


  PetscFunctionReturn(0);
}
