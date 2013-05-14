#include "petiga.h"
#if PETSC_VERSION_LE(3,2,0)
#include <private/pcimpl.h>
#else
#include <petsc-private/pcimpl.h>
#endif
#include "petigabl.h"

#if PETSC_VERSION_LE(3,3,0)
#undef MatType
typedef const char* MatType;
#endif

typedef struct {
  Mat mat;
} PC_EBE;

#undef  __FUNCT__
#define __FUNCT__ "PCSetUp_EBE_CreateMatrix"
static PetscErrorCode PCSetUp_EBE_CreateMatrix(Mat A, Mat *B)
{
  MPI_Comm       comm = ((PetscObject)A)->comm;
  PetscMPIInt    size;
  Mat            mat = 0;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) {
    void (*aij)(void) = 0;
    void (*baij)(void) = 0;
    void (*sbaij)(void) = 0;
    ierr = PetscObjectQueryFunction((PetscObject)A,"MatMPIAIJSetPreallocation_C",&aij);CHKERRQ(ierr);
    ierr = PetscObjectQueryFunction((PetscObject)A,"MatMPIBAIJSetPreallocation_C",&baij);CHKERRQ(ierr);
    ierr = PetscObjectQueryFunction((PetscObject)A,"MatMPISBAIJSetPreallocation_C",&sbaij);CHKERRQ(ierr);
    if (aij || baij || sbaij) {
      Mat Ad = 0;
      ierr = PetscTryMethod(A,"MatGetDiagonalBlock_C",(Mat,Mat*),(A,&Ad));CHKERRQ(ierr);
      if (Ad) {
        PetscBool compressed,done;
        PetscInt  na;
        #if PETSC_VERSION_LE(3,3,0)
        PetscInt *ia,*ja;
        #else
        const PetscInt *ia,*ja;
        #endif
        ierr = MatGetDiagonalBlock(A,&Ad);CHKERRQ(ierr);
        compressed = (baij||sbaij) ? PETSC_TRUE: PETSC_FALSE;
        ierr = MatGetRowIJ(Ad,0,PETSC_FALSE,compressed,&na,&ia,&ja,&done);CHKERRQ(ierr);
        if (done) {
          PetscInt m,n,M,N,bs;
          PetscInt j,cstart,*newja;
          MatType mtype;
          ierr = MatGetType(A,&mtype);CHKERRQ(ierr);
          ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
          ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
          ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);
          ierr = MatGetOwnershipRangeColumn(A,&cstart,NULL);CHKERRQ(ierr);
          if (baij || sbaij) cstart /= bs;

          ierr = MatCreate(comm,&mat);CHKERRQ(ierr);
          ierr = MatSetType(mat,mtype);CHKERRQ(ierr);
          ierr = MatSetSizes(mat,m,n,M,N);CHKERRQ(ierr);
          #if PETSC_VERSION_GT(3,2,0)
          ierr = MatSetBlockSize(mat,bs);CHKERRQ(ierr);
          #endif

          ierr = PetscMalloc1(ia[na],PetscInt,&newja);CHKERRQ(ierr);
          for (j=0; j<ia[na]; j++) newja[j] = ja[j] + cstart;
          if (aij)   {ierr = MatMPIAIJSetPreallocationCSR  (mat,   ia,newja,NULL);CHKERRQ(ierr);}
          if (baij)  {ierr = MatMPIBAIJSetPreallocationCSR (mat,bs,ia,newja,NULL);CHKERRQ(ierr);}
          if (sbaij) {ierr = MatMPISBAIJSetPreallocationCSR(mat,bs,ia,newja,NULL);CHKERRQ(ierr);}
          ierr = PetscFree(newja);CHKERRQ(ierr);
          #if PETSC_VERSION_(3,2,0)
          ierr = MatSetBlockSize(mat,bs);CHKERRQ(ierr);
          #endif
        }
        ierr = MatRestoreRowIJ(Ad,0,PETSC_FALSE,compressed,&na,&ia,&ja,&done);CHKERRQ(ierr);
      }
    }
  }
  if (!mat) {
    #if PETSC_VERSION_(3,2,0)
    ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&mat);CHKERRQ(ierr);
    #else
    ierr = MatDuplicate(A,MAT_SHARE_NONZERO_PATTERN,&mat);CHKERRQ(ierr);
    #endif
  }
  ierr = MatSetOption(mat,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  *B = mat;
  PetscFunctionReturn(0);
}

static PetscInt ComputeOwnedGlobalIndices(const PetscInt lgmap[],
                                          PetscInt start, PetscInt end,PetscInt bs,
                                          PetscInt N, const PetscInt idx[], PetscInt idxout[])
{
  PetscInt i,c,Nout=0;
  for (i=0; i<N; i++) {
    PetscInt index = lgmap[idx[i]];
    if (index >= start && index < end)
      for (c=0; c<bs; c++)
        idxout[Nout++] = c + index*bs;
  }
  return Nout;
}

#undef  __FUNCT__
#define __FUNCT__ "PCSetUp_EBE"
static PetscErrorCode PCSetUp_EBE(PC pc)
{
  PC_EBE         *ebe = (PC_EBE*)pc->data;
  IGA            iga = 0;
  Mat            A,B;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  A = pc->pmat;
  ierr = PetscObjectQuery((PetscObject)A,"IGA",(PetscObject*)&iga);CHKERRQ(ierr);
  if (!iga) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Matrix is missing the IGA context");
  PetscValidHeaderSpecific(iga,IGA_CLASSID,0);

  if (pc->flag != SAME_NONZERO_PATTERN) {
    ierr = MatDestroy(&ebe->mat);CHKERRQ(ierr);
  }
  if (!ebe->mat) {
    ierr = PCSetUp_EBE_CreateMatrix(A,&ebe->mat);CHKERRQ(ierr);
  }
  B = ebe->mat;

  ierr = MatZeroEntries(B);CHKERRQ(ierr);
  {
    IGAElement   element;
    PetscInt     nen,dof;
    PetscInt     n,*indices;
    PetscScalar  *values,*work,lwkopt;
    PetscBLASInt m,*ipiv,info,lwork;
    PetscInt     start,end;
    const PetscInt *ltogmap;
    const PetscInt *mapping;
    ISLocalToGlobalMapping map;

    ierr = IGAGetElement(iga,&element);CHKERRQ(ierr);
    ierr = IGAElementGetSizes(element,NULL,&nen,&dof);CHKERRQ(ierr);

    if (dof == 1) {
      ierr = MatGetLocalToGlobalMapping(A,&map,NULL);CHKERRQ(ierr);
    } else {
      ierr = MatGetLocalToGlobalMappingBlock(A,&map,NULL);CHKERRQ(ierr);
    }
    ierr = ISLocalToGlobalMappingGetIndices(map,&ltogmap);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(A,&start,&end);CHKERRQ(ierr);
    start /= dof; end /= dof;

    n = nen*dof;
    ierr = PetscBLASIntCast(n,&m);CHKERRQ(ierr);
    ierr = PetscMalloc2(n,PetscInt,&indices,n*n,PetscScalar,&values);CHKERRQ(ierr);
    ierr = PetscMalloc1(m,PetscBLASInt,&ipiv);CHKERRQ(ierr);
    lwork = -1; work = &lwkopt;
    LAPACKgetri_(&m,values,&m,ipiv,work,&lwork,&info);
    lwork = (info==0) ? (PetscBLASInt)work[0] : m*128;
    ierr = PetscMalloc1(lwork,PetscScalar,&work);CHKERRQ(ierr);

    ierr = IGABeginElement(iga,&element);CHKERRQ(ierr);
    while (IGANextElement(iga,element)) {
      ierr = IGAElementGetMapping(element,&nen,&mapping);CHKERRQ(ierr);
      m = n = ComputeOwnedGlobalIndices(ltogmap,start,end,dof,nen,mapping,indices);
      /* get element matrix from global matrix */
      ierr = MatGetValues(A,n,indices,n,indices,values);CHKERRQ(ierr);
      /* compute inverse of element matrix */
      LAPACKgetrf_(&m,&m,values,&m,ipiv,&info);
      if (info<0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Bad argument to LAPACKgetrf_");
      if (info>0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Zero-pivot in LU factorization");
      ierr = PetscLogFlops((1/3.*n*n*n         +2/3.*n));CHKERRQ(ierr); /* multiplications */
      ierr = PetscLogFlops((1/3.*n*n*n-1/2.*n*n+1/6.*n));CHKERRQ(ierr); /* additions */
      LAPACKgetri_(&m,values,&m,ipiv,work,&lwork,&info);
      if (info<0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Bad argument to LAPACKgetri_");
      if (info>0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Zero-pivot in LU factorization");
      ierr = PetscLogFlops((2/3.*n*n*n+1/2.*n*n+5/6.*n));CHKERRQ(ierr); /* multiplications */
      ierr = PetscLogFlops((2/3.*n*n*n-3/2.*n*n+5/6.*n));CHKERRQ(ierr); /* additions */
      /* add values back into preconditioner matrix */
      ierr = MatSetValues(B,n,indices,n,indices,values,ADD_VALUES);CHKERRQ(ierr);
      ierr = PetscLogFlops(n*n);CHKERRQ(ierr);
    }
    ierr = IGAEndElement(iga,&element);CHKERRQ(ierr);

    ierr = ISLocalToGlobalMappingRestoreIndices(map,&ltogmap);CHKERRQ(ierr);
    ierr = PetscFree2(indices,values);CHKERRQ(ierr);
    ierr = PetscFree(ipiv);CHKERRQ(ierr);
    ierr = PetscFree(work);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*
#undef  __FUNCT__
#define __FUNCT__ "PCSetFromOptions_EBE"
static PetscErrorCode PCSetFromOptions_EBE(PC pc)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
*/

#undef  __FUNCT__
#define __FUNCT__ "PCApply_EBE"
static PetscErrorCode PCApply_EBE(PC pc, Vec x,Vec y)
{
  PC_EBE         *ebe = (PC_EBE*)pc->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MatMult(ebe->mat,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "PCApplyTranspose_EBE"
static PetscErrorCode PCApplyTranspose_EBE(PC pc, Vec x,Vec y)
{
  PC_EBE         *ebe = (PC_EBE*)pc->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MatMultTranspose(ebe->mat,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "PCView_EBE"
static PetscErrorCode PCView_EBE(PC pc,PetscViewer viewer)
{
  PC_EBE         *ebe = (PC_EBE*)pc->data;
  PetscBool      isascii;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (!isascii) PetscFunctionReturn(0);
  if (!ebe->mat) PetscFunctionReturn(0);
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"element-by-element matrix:\n");CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
  ierr = MatView(ebe->mat,viewer);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "PCReset_EBE"
static PetscErrorCode PCReset_EBE(PC pc)
{
  PC_EBE         *ebe = (PC_EBE*)pc->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MatDestroy(&ebe->mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "PCDestroy_EBE"
static PetscErrorCode PCDestroy_EBE(PC pc)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PCReset_EBE(pc);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


EXTERN_C_BEGIN
#undef  __FUNCT__
#define __FUNCT__ "PCCreate_IGAEBE"
PetscErrorCode PCCreate_IGAEBE(PC pc)
{
  PC_EBE         *ebe = 0;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscNewLog(pc,PC_EBE,&ebe);CHKERRQ(ierr);
  pc->data = (void*)ebe;

  pc->ops->setup               = PCSetUp_EBE;
  pc->ops->reset               = PCReset_EBE;
  pc->ops->destroy             = PCDestroy_EBE;
  pc->ops->setfromoptions      = 0;/* PCSetFromOptions_EBE; */
  pc->ops->view                = PCView_EBE;
  pc->ops->apply               = PCApply_EBE;
  pc->ops->applytranspose      = PCApplyTranspose_EBE;

  PetscFunctionReturn(0);
}
EXTERN_C_END
