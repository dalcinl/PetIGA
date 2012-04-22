#include "petiga.h"

#if PETSC_VERSION_(3,2,0)
#include "private/matimpl.h"
#endif

EXTERN_C_BEGIN
extern PetscErrorCode MatView_MPI_DA(Mat,PetscViewer);
extern PetscErrorCode MatLoad_MPI_DA(Mat,PetscViewer);
EXTERN_C_END

PETSC_STATIC_INLINE
PetscInt Product(const PetscInt a[3]) { return a[0]*a[1]*a[2]; }

PETSC_STATIC_INLINE
PetscInt Index3D(const PetscInt start[3],const PetscInt shape[3],
                 PetscInt i,PetscInt j,PetscInt k)
{
  if (start) { i -= start[0]; j -= start[1]; k -= start[2]; }
  return i + j * shape[0] + k * shape[0] * shape[1];
}

PETSC_STATIC_INLINE
void BasisStencil(IGA iga,PetscInt dir,PetscInt i,PetscInt *first,PetscInt *last)
{
  PetscInt p  = iga->axis[dir]->p;
  const PetscReal *U = iga->axis[dir]->U;
  PetscInt k;
  /* compute index of the leftmost overlapping basis */
  k = i;
  while (U[k]==U[k+1]) k++; /* XXX Using "==" with floating point values ! */
  *first = k - p;
  /* cmopute index of the rightmost overlapping basis */
  k = i + p + 1;
  while (U[k]==U[k-1]) k--; /* XXX Using "==" with floating point values ! */
  *last = k - 1;
}

PETSC_STATIC_INLINE
PetscInt ColumnIndices(IGA iga,const PetscInt start[3],const PetscInt shape[3],
                       PetscInt iA,PetscInt jA,PetscInt kA,PetscInt *stencil)
{
  PetscInt first[3] = {0,0,0};
  PetscInt last [3] = {0,0,0};
  PetscInt count    = 0;
  { /* compute range of overlapping basis in each direction */
    PetscInt i,A[3]; A[0] = iA; A[1] = jA; A[2] = kA;
    for (i=0; i<iga->dim; i++)
      BasisStencil(iga,i,A[i],&first[i],&last[i]);
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
  PetscInt       i,n;
  PetscFunctionBegin;
  for (i=0,n=0; i<*cnt; i++)
    if (col[i] >= row)
      col[n++] = col[i];
  *cnt = n;
  PetscFunctionReturn(0);
}

extern PetscErrorCode IGA_Grid_CreateLGMap(MPI_Comm,PetscInt,PetscInt,const PetscInt[],const PetscInt[],const PetscInt[],AO,LGMap*);

#undef  __FUNCT__
#define __FUNCT__ "IGACreateMat"
PetscErrorCode IGACreateMat(IGA iga,Mat *mat)
{
  MPI_Comm       comm;
  PetscMPIInt    size;
  PetscBool      aij,baij,sbaij;
  PetscInt       i,dim,*sizes,*start,*width;
  PetscInt       gstart[3] = {0,0,0};
  PetscInt       gwidth[3] = {1,1,1};
  PetscInt       maxnnz;
  PetscInt       n,N,bs;
  AO             aob;
  LGMap          ltog,ltogb;
  const MatType  mtype;
  Mat            A;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(mat,2);
  IGACheckSetUp(iga,1);

  ierr = PetscObjectGetComm((PetscObject)iga,&comm);CHKERRQ(ierr);
  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
  ierr = IGAGetDof(iga,&bs);CHKERRQ(ierr);
  mtype = iga->mattype;
  aob = iga->aob;

  sizes = iga->node_sizes;
  start = iga->node_start;
  width = iga->node_width;
  for (i=0; i<dim; i++) {
    PetscInt first = start[i];
    PetscInt last  = first + width[i] - 1;
    PetscInt gfirst,glast;
    BasisStencil(iga,i,first,&gstart[i],&glast);
    BasisStencil(iga,i,last,&gfirst,&glast);
    gwidth[i] = glast + 1 - gstart[i];
  }
  maxnnz = 1;
  for(i=0; i<dim; i++)
    maxnnz *= (2*iga->axis[i]->p + 1); /* XXX do better ? */

  n = Product(width);
  N = Product(sizes);

  ierr = MatCreate(comm,&A);CHKERRQ(ierr);
#if PETSC_VERSION_(3,2,0)
  ierr = MatSetType(A,mtype);CHKERRQ(ierr);
  ierr = MatSetSizes(A,bs*n,bs*n,bs*N,bs*N);CHKERRQ(ierr);
#else
  ierr = MatSetSizes(A,bs*n,bs*n,bs*N,bs*N);CHKERRQ(ierr);
  ierr = MatSetBlockSize(A,bs);CHKERRQ(ierr);
  ierr = MatSetType(A,mtype);CHKERRQ(ierr);
#endif
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = InferMatrixType(A,&aij,&baij,&sbaij);CHKERRQ(ierr);

  ierr = IGA_Grid_CreateLGMap(comm,dim,1,sizes,gstart,gwidth,aob,&ltogb);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingUnBlock(ltogb,bs,&ltog);CHKERRQ(ierr);
  if (aij || baij || sbaij) {
    PetscInt nbs = (baij||sbaij) ? n : n*bs;
    PetscInt *dnz = 0, *onz = 0;
    ierr = MatPreallocateInitialize(comm,nbs,nbs,dnz,onz);CHKERRQ(ierr);
    {
      PetscInt i,j,k;
      PetscInt n = maxnnz,*indices=0,*ubrows,*ubcols=0;
      ierr = PetscMalloc1(n,PetscInt,&indices);CHKERRQ(ierr);
      ierr = PetscMalloc2(bs,PetscInt,&ubrows,n*bs,PetscInt,&ubcols);CHKERRQ(ierr);
      for (k=start[2]; k<start[2]+width[2]; k++)
        for (j=start[1]; j<start[1]+width[1]; j++)
          for (i=start[0]; i<start[0]+width[0]; i++)
            { /* */
              PetscInt row   = Index3D(gstart,gwidth,i,j,k);
              PetscInt count = ColumnIndices(iga,gstart,gwidth,i,j,k,indices);
              if (aij) {
                if (bs == 1) {
                  ierr = MatPreallocateSetLocal(ltog,1,&row,ltog,count,indices,dnz,onz);CHKERRQ(ierr);
                } else {
                  ierr = UnblockIndices(bs,row,count,indices,ubrows,ubcols);CHKERRQ(ierr);
                  ierr = MatPreallocateSetLocal(ltog,bs,ubrows,ltog,count*bs,ubcols,dnz,onz);CHKERRQ(ierr);
                }
              } else if (baij) {
                ierr = MatPreallocateSetLocal(ltogb,1,&row,ltogb,count,indices,dnz,onz);CHKERRQ(ierr);
              } else if (sbaij) {
                ierr = L2GApply(ltogb,&row,&count,indices);CHKERRQ(ierr);
                ierr = FilterLowerTriangular(row,&count,indices);CHKERRQ(ierr);
                ierr = MatPreallocateSymmetricSet(row,count,indices,dnz,onz);CHKERRQ(ierr);
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
      ierr = PetscFree2(ubrows,ubcols);CHKERRQ(ierr);
      ierr = PetscFree(indices);CHKERRQ(ierr);
    }
    ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  } else {
    ierr = MatSetUp(A);CHKERRQ(ierr);
  }
#if PETSC_VERSION_(3,2,0)
  /* XXX This is a vile hack. Perhaps we should just check for      */
  /* SeqDense and MPIDense that are the only I care about right now */
  if (A->ops->setblocksize) {
    ierr = MatSetBlockSize(A,bs);CHKERRQ(ierr);
  } else {
    ierr = PetscLayoutSetBlockSize(A->rmap,bs);CHKERRQ(ierr);
    ierr = PetscLayoutSetBlockSize(A->cmap,bs);CHKERRQ(ierr);
  }
#endif
  if (aij || baij || sbaij) {
    PetscInt i,j,k;
    PetscInt n = maxnnz,*indices=0,*ubrows,*ubcols=0;PetscScalar *values=0;
    ierr = PetscMalloc2(n,PetscInt,&indices,n*bs*n*bs,PetscScalar,&values);CHKERRQ(ierr);
    ierr = PetscMalloc2(bs,PetscInt,&ubrows,n*bs,PetscInt,&ubcols);CHKERRQ(ierr);
    ierr = PetscMemzero(values,n*bs*n*bs*sizeof(PetscScalar));CHKERRQ(ierr);
    for (k=start[2]; k<start[2]+width[2]; k++)
      for (j=start[1]; j<start[1]+width[1]; j++)
        for (i=start[0]; i<start[0]+width[0]; i++)
          { /* */
            PetscInt row   = Index3D(gstart,gwidth,i,j,k);
            PetscInt count = ColumnIndices(iga,gstart,gwidth,i,j,k,indices);
            ierr = L2GApply(ltogb,&row,&count,indices);CHKERRQ(ierr);
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
  ierr = ISLocalToGlobalMappingDestroy(&ltog);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&ltogb);CHKERRQ(ierr);

  ierr = MatSetLocalToGlobalMapping(A,iga->lgmap,iga->lgmap);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMappingBlock(A,iga->lgmapb,iga->lgmapb);CHKERRQ(ierr);

  ierr = PetscObjectCompose((PetscObject)A,"IGA",(PetscObject)iga);CHKERRQ(ierr);
  *mat = A;

  { /* XXX */
    ierr = PetscObjectCompose((PetscObject)*mat,"DM",(PetscObject)iga->dm_dof);CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
    if (size > 1) { /* change viewer to display matrix in natural ordering */
      ierr = MatShellSetOperation(*mat,MATOP_VIEW,(void (*)(void))MatView_MPI_DA);CHKERRQ(ierr);
      ierr = MatShellSetOperation(*mat,MATOP_LOAD,(void (*)(void))MatLoad_MPI_DA);CHKERRQ(ierr);
    }
  } /* XXX */
  PetscFunctionReturn(0);
}
