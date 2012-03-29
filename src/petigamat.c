#include "petiga.h"

#if PETSC_VERSION_(3,2,0)
#include "private/matimpl.h"
#endif

EXTERN_C_BEGIN
extern PetscErrorCode MatView_MPI_DA(Mat,PetscViewer);
extern PetscErrorCode MatLoad_MPI_DA(Mat,PetscViewer);
EXTERN_C_BEGIN

PETSC_STATIC_INLINE
PetscInt Product(const PetscInt a[3]) { return a[0]*a[1]*a[2]; }

PETSC_STATIC_INLINE
PetscInt Index3D(const PetscInt start[3],const PetscInt shape[3],PetscInt i,PetscInt j,PetscInt k)
{
  if (start) { i -= start[0]; j -= start[1]; k -= start[2]; }
  return i + j * shape[0] + k * shape[0] * shape[1];
}

PETSC_STATIC_INLINE
void BasisRange1D(PetscInt i,PetscInt p,const PetscReal U[],PetscInt *first,PetscInt *last)
{
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
PetscInt BuildIndices(IGA iga,PetscInt iA,PetscInt jA,PetscInt kA,PetscInt *indices)
{
  PetscInt first[3] = {0,0,0};
  PetscInt last [3] = {0,0,0};
  PetscInt count    = 0;
  { /* compute range of overlapping basis in each direction */
    IGAAxis  *axis = iga->axis;
    PetscInt i,A[3]; A[0] = iA; A[1] = jA; A[2] = kA;
    for (i=0; i<iga->dim; i++)
      BasisRange1D(A[i],axis[i]->p,axis[i]->U,&first[i],&last[i]);
  }
  { /* tensor-product the ranges of overlapping basis */
    PetscInt i,j,k;
    PetscInt *start = iga->ghost_start;
    PetscInt *shape = iga->ghost_width;
    for (k=first[2]; k<=last[2]; k++)
    for (j=first[1]; j<=last[1]; j++)
    for (i=first[0]; i<=last[0]; i++)
    indices[count++] = Index3D(start,shape,i,j,k);
  }
  return count;
}

PETSC_STATIC_INLINE
PetscInt UnblockIndices(PetscInt bs,PetscInt row,PetscInt count,const PetscInt indices[],PetscInt ubrows[],PetscInt ubcols[])
{
  PetscInt n,c;
  for (c=0; c<bs; c++)
    ubrows[c] = c + row*bs;
  for (n=0; n<count; n++)
    for (c=0; c<bs; c++)
      ubcols[c+n*bs] = c + indices[n]*bs;
  return 0;
}

PETSC_STATIC_INLINE
#undef  __FUNCT__
#define __FUNCT__ "GetMatrixTraits"
PetscErrorCode GetMatrixTraits(Mat A,PetscBool *aij,PetscBool *baij,PetscBool *sbaij)
{
  void (*f)(void) = 0;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  *aij = *baij = *sbaij = PETSC_FALSE;
  if (!f) {ierr = PetscObjectQueryFunction((PetscObject)A,"MatMPIAIJSetPreallocation_C",&f);CHKERRQ(ierr);}
  if (!f) {ierr = PetscObjectQueryFunction((PetscObject)A,"MatSeqAIJSetPreallocation_C",&f);CHKERRQ(ierr);}
  if ( f) goto is_aij;
  if (!f) {ierr = PetscObjectQueryFunction((PetscObject)A,"MatMPIBAIJSetPreallocation_C",&f);CHKERRQ(ierr);}
  if (!f) {ierr = PetscObjectQueryFunction((PetscObject)A,"MatSeqBAIJSetPreallocation_C",&f);CHKERRQ(ierr);}
  if ( f) goto is_baij;
  if (!f) {ierr = PetscObjectQueryFunction((PetscObject)A,"MatMPISBAIJSetPreallocation_C",&f);CHKERRQ(ierr);}
  if (!f) {ierr = PetscObjectQueryFunction((PetscObject)A,"MatSeqSBAIJSetPreallocation_C",&f);CHKERRQ(ierr);}
  if ( f) goto is_sbaij;
  PetscFunctionReturn(0);
 is_sbaij: *sbaij = PETSC_TRUE;
 is_baij:  *baij  = PETSC_TRUE;
 is_aij:   *aij   = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE
#undef __FUNCT__
#define __FUNCT__ "L2GFilterUpperTriangular"
/*
  This helper is for of SBAIJ preallocation, to discard the lower-triangular values
  which are difficult to identify in the local ordering with periodic domain.
*/
PetscErrorCode L2GFilterUpperTriangular(ISLocalToGlobalMapping ltog,PetscInt *row,PetscInt *cnt,PetscInt col[])
{
  PetscInt       i,n;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = ISLocalToGlobalMappingApply(ltog,1,row,row);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApply(ltog,*cnt,col,col);CHKERRQ(ierr);
  for (i=0,n=0; i<*cnt; i++)
    if (col[i] >= *row)
      col[n++] = col[i];
  *cnt = n;
  PetscFunctionReturn(0);
}

#define ISLGMap ISLocalToGlobalMapping

#undef  __FUNCT__
#define __FUNCT__ "IGACreateMat"
PetscErrorCode IGACreateMat(IGA iga,Mat *mat)
{
  MPI_Comm       comm;
  PetscMPIInt    size;
  PetscInt       n,N,bs;
  PetscBool      aij,baij,sbaij;
  ISLGMap        ltog,ltogb;
  DM             dm;
  Mat            A;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidPointer(mat,2);
  IGACheckSetUp(iga,1);

  ierr = PetscObjectGetComm((PetscObject)iga,&comm);CHKERRQ(ierr);

  ierr = IGAGetDof(iga,&bs);CHKERRQ(ierr);
  n = Product(iga->node_width);
  N = Product(iga->node_sizes);

  ierr = MatCreate(comm,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,iga->mattype);CHKERRQ(ierr);
  ierr = MatSetSizes(A,bs*n,bs*n,bs*N,bs*N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);

  ierr = IGAGetDofDM(iga,&dm);CHKERRQ(ierr);
  ierr = MatSetDM(A,dm);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(dm,&ltog);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMappingBlock(dm,&ltogb);CHKERRQ(ierr);

  ierr = GetMatrixTraits(A,&aij,&baij,&sbaij);CHKERRQ(ierr);

  if (sbaij) {
    PetscInt nbs = n, *dnz = 0, *onz = 0;
    ierr = MatPreallocateSymmetricInitialize(comm,nbs,nbs,dnz,onz);CHKERRQ(ierr);
    {
      PetscInt i,j,k;
      PetscInt *start = iga->node_start, *ghost_start = iga->ghost_start;
      PetscInt *width = iga->node_width, *ghost_width = iga->ghost_width;
      PetscInt n=1,*indices=0;
      for(i=0; i<iga->dim; i++) n *= 2*iga->axis[i]->p + 1; /* XXX do better ? */
      ierr = PetscMalloc1(n,PetscInt,&indices);CHKERRQ(ierr);
      for (k=start[2]; k<start[2]+width[2]; k++)
        for (j=start[1]; j<start[1]+width[1]; j++)
          for (i=start[0]; i<start[0]+width[0]; i++)
            { /* */
              PetscInt row = Index3D(ghost_start,ghost_width,i,j,k);
              PetscInt count = BuildIndices(iga,i,j,k,indices);
              ierr = L2GFilterUpperTriangular(ltogb,&row,&count,indices);CHKERRQ(ierr);
              ierr = MatPreallocateSymmetricSet(row,count,indices,dnz,onz);CHKERRQ(ierr);
            }
      ierr = PetscFree(indices);CHKERRQ(ierr);
      ierr = MatSeqSBAIJSetPreallocation(A,bs,0,dnz);CHKERRQ(ierr);
      ierr = MatMPISBAIJSetPreallocation(A,bs,0,dnz,0,onz);CHKERRQ(ierr);
    }
    ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
  } else if (aij || baij) {
    PetscInt nbs = baij ? n : n*bs, *dnz = 0, *onz = 0;
    ierr = MatPreallocateInitialize(comm,nbs,nbs,dnz,onz);CHKERRQ(ierr);
    {
      PetscInt i,j,k;
      PetscInt *start = iga->node_start, *ghost_start = iga->ghost_start;
      PetscInt *width = iga->node_width, *ghost_width = iga->ghost_width;
      PetscInt n=1,*indices=0,*ubrows,*ubcols=0;
      for(i=0; i<iga->dim; i++) n *= 2*iga->axis[i]->p + 1; /* XXX do better ? */
      ierr = PetscMalloc1(n,PetscInt,&indices);CHKERRQ(ierr);
      ierr = PetscMalloc2(bs,PetscInt,&ubrows,n*bs,PetscInt,&ubcols);CHKERRQ(ierr);
      for (k=start[2]; k<start[2]+width[2]; k++)
        for (j=start[1]; j<start[1]+width[1]; j++)
          for (i=start[0]; i<start[0]+width[0]; i++)
            { /* */
              PetscInt row = Index3D(ghost_start,ghost_width,i,j,k);
              PetscInt count = BuildIndices(iga,i,j,k,indices);
              if (baij || bs == 1) {
                ierr = MatPreallocateSetLocal(ltogb,1,&row,ltogb,count,indices,dnz,onz);CHKERRQ(ierr);
              } else  {
                ierr = UnblockIndices(bs,row,count,indices,ubrows,ubcols);CHKERRQ(ierr);
                ierr = MatPreallocateSetLocal(ltog,bs,ubrows,ltog,count*bs,ubcols,dnz,onz);CHKERRQ(ierr);
              }
            }
      ierr = PetscFree2(ubrows,ubcols);CHKERRQ(ierr);
      ierr = PetscFree(indices);CHKERRQ(ierr);
      if (baij) {
        ierr = MatSeqBAIJSetPreallocation(A,bs,0,dnz);CHKERRQ(ierr);
        ierr = MatMPIBAIJSetPreallocation(A,bs,0,dnz,0,onz);CHKERRQ(ierr);
      } else {
        ierr = MatSeqAIJSetPreallocation(A,0,dnz);CHKERRQ(ierr);
        ierr = MatMPIAIJSetPreallocation(A,0,dnz,0,onz);CHKERRQ(ierr);
      }
    }
    ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
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
#else
  ierr = MatSetBlockSize(A,bs);CHKERRQ(ierr);
#endif

  ierr = MatSetLocalToGlobalMapping(A,ltog,ltog);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMappingBlock(A,ltogb,ltogb);CHKERRQ(ierr);

  if (aij || baij || sbaij) {
    PetscInt i,j,k;
    PetscInt *start = iga->node_start, *ghost_start = iga->ghost_start;
    PetscInt *width = iga->node_width, *ghost_width = iga->ghost_width;
    PetscInt n=1,*indices=0,*ubrows=0,*ubcols=0;
    PetscScalar *values=0;
    for(i=0; i<iga->dim; i++) n *= 2*iga->axis[i]->p + 1; /* XXX do better ? */
    ierr = PetscMalloc2(n,PetscInt,&indices,n*bs*n*bs,PetscScalar,&values);CHKERRQ(ierr);
    ierr = PetscMemzero(values,n*bs*n*bs*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PetscMalloc2(bs,PetscInt,&ubrows,n*bs,PetscInt,&ubcols);CHKERRQ(ierr);
    for (k=start[2]; k<start[2]+width[2]; k++)
      for (j=start[1]; j<start[1]+width[1]; j++)
        for (i=start[0]; i<start[0]+width[0]; i++)
          { /* */
            PetscInt row = Index3D(ghost_start,ghost_width,i,j,k);
            PetscInt count = BuildIndices(iga,i,j,k,indices);
            if (sbaij) {
              ierr = L2GFilterUpperTriangular(ltogb,&row,&count,indices);CHKERRQ(ierr);
              ierr = MatSetValuesBlocked(A,1,&row,count,indices,values,INSERT_VALUES);CHKERRQ(ierr);
            } else if (baij) {
              ierr = MatSetValuesBlockedLocal(A,1,&row,count,indices,values,INSERT_VALUES);CHKERRQ(ierr);
            } else /* (aij) */ {
              if (bs == 1) {
                ierr = MatSetValuesLocal(A,1,&row,count,indices,values,INSERT_VALUES);CHKERRQ(ierr);
              } else {
                ierr = UnblockIndices(bs,row,count,indices,ubrows,ubcols);CHKERRQ(ierr);
                ierr = MatSetValuesLocal(A,bs,ubrows,count*bs,ubcols,values,INSERT_VALUES);CHKERRQ(ierr);
              }
            }
          }
    ierr = PetscFree2(ubrows,ubcols);CHKERRQ(ierr);
    ierr = PetscFree2(indices,values);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd  (A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  ierr = PetscObjectCompose((PetscObject)A,"DM",(PetscObject)dm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) { /* change viewer to display matrix in natural ordering */
    ierr = MatShellSetOperation(A,MATOP_VIEW,(void (*)(void))MatView_MPI_DA);CHKERRQ(ierr);
    ierr = MatShellSetOperation(A,MATOP_LOAD,(void (*)(void))MatLoad_MPI_DA);CHKERRQ(ierr);
  }

  ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  /*ierr = MatSetOption(A,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE);CHKERRQ(ierr);*/
  /*ierr = MatSetOption(A,MAT_STRUCTURALLY_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);*/
  ierr = PetscObjectCompose((PetscObject)A,"IGA",(PetscObject)iga);CHKERRQ(ierr);

  *mat = A;
  PetscFunctionReturn(0);
}
