#include "petiga.h"
#include <petscblaslapack.h>
#include <petsc/private/pcimpl.h>

typedef struct {
  Mat mat;
} PC_EBE;

static PetscErrorCode PCSetUp_EBE_CreateMatrix(Mat A,Mat *B)
{
  MPI_Comm       comm = ((PetscObject)A)->comm;
  PetscMPIInt    size;
  Mat            mat = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) {
    void (*aij)(void) = NULL;
    void (*baij)(void) = NULL;
    void (*sbaij)(void) = NULL;
    ierr = PetscObjectQueryFunction((PetscObject)A,"MatMPIAIJSetPreallocation_C",&aij);CHKERRQ(ierr);
    ierr = PetscObjectQueryFunction((PetscObject)A,"MatMPIBAIJSetPreallocation_C",&baij);CHKERRQ(ierr);
    ierr = PetscObjectQueryFunction((PetscObject)A,"MatMPISBAIJSetPreallocation_C",&sbaij);CHKERRQ(ierr);
    if (aij || baij || sbaij) {
      Mat Ad = NULL;
      ierr = PetscTryMethod(A,"MatGetDiagonalBlock_C",(Mat,Mat*),(A,&Ad));CHKERRQ(ierr);
      if (Ad) {
        PetscInt na;
        const PetscInt *ia,*ja;
        PetscBool compressed,done;
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
          ierr = MatSetBlockSize(mat,bs);CHKERRQ(ierr);

          ierr = PetscMalloc1((size_t)ia[na],&newja);CHKERRQ(ierr);
          for (j=0; j<ia[na]; j++) newja[j] = ja[j] + cstart;
          if (aij)   {ierr = MatMPIAIJSetPreallocationCSR(mat,ia,newja,NULL);CHKERRQ(ierr);}
          if (baij)  {ierr = MatMPIBAIJSetPreallocationCSR(mat,bs,ia,newja,NULL);CHKERRQ(ierr);}
          if (sbaij) {ierr = MatMPISBAIJSetPreallocationCSR(mat,bs,ia,newja,NULL);CHKERRQ(ierr);}
          ierr = PetscFree(newja);CHKERRQ(ierr);
        }
        ierr = MatRestoreRowIJ(Ad,0,PETSC_FALSE,compressed,&na,&ia,&ja,&done);CHKERRQ(ierr);
      }
    }
  }
  if (!mat) {ierr = MatDuplicate(A,MAT_SHARE_NONZERO_PATTERN,&mat);CHKERRQ(ierr);}
  ierr = MatSetOption(mat,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  *B = mat;
  PetscFunctionReturn(0);
}

static PetscInt ComputeOwnedGlobalIndices(const PetscInt lgmap[],PetscInt bs,
                                          PetscInt start,PetscInt end,
                                          PetscInt N,const PetscInt idx[],PetscInt idxout[])
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

static PetscErrorCode PCSetUp_EBE(PC pc)
{
  PC_EBE         *ebe = (PC_EBE*)pc->data;
  IGA            iga = NULL;
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

    ierr = MatGetLocalToGlobalMapping(A,&map,NULL);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetBlockIndices(map,&ltogmap);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(A,&start,&end);CHKERRQ(ierr);
    start /= dof; end /= dof;

    n = nen*dof;
    ierr = PetscBLASIntCast(n,&m);CHKERRQ(ierr);
    ierr = PetscMalloc1((size_t)n,&indices);CHKERRQ(ierr);
    ierr = PetscMalloc1((size_t)n*(size_t)n,&values);CHKERRQ(ierr);
    ierr = PetscMalloc1((size_t)m,&ipiv);CHKERRQ(ierr);
    lwork = -1; work = &lwkopt;
    LAPACKgetri_(&m,values,&m,ipiv,work,&lwork,&info);
    lwork = (info==0) ? (PetscBLASInt)PetscRealPart(work[0]) : m*128;
    ierr = PetscMalloc1((size_t)lwork,&work);CHKERRQ(ierr);

    ierr = IGABeginElement(iga,&element);CHKERRQ(ierr);
    while (IGANextElement(iga,element)) {
      ierr = IGAElementGetClosure(element,&nen,&mapping);CHKERRQ(ierr);
      n = ComputeOwnedGlobalIndices(ltogmap,dof,start,end,nen,mapping,indices);
      ierr = PetscBLASIntCast(n,&m);CHKERRQ(ierr);
      /* get element matrix from global matrix */
      ierr = MatGetValues(A,n,indices,n,indices,values);CHKERRQ(ierr);
      /* compute inverse of element matrix */
      LAPACKgetrf_(&m,&m,values,&m,ipiv,&info);
      if (info<0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Bad argument to LAPACKgetrf_");
      if (info>0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Zero-pivot in LU factorization");
      ierr = PetscLogFlops((PetscLogDouble)(1*n*n*n      +2*n)/3);CHKERRQ(ierr); /* multiplications */
      ierr = PetscLogFlops((PetscLogDouble)(2*n*n*n-3*n*n+1*n)/6);CHKERRQ(ierr); /* additions */
      LAPACKgetri_(&m,values,&m,ipiv,work,&lwork,&info);
      if (info<0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Bad argument to LAPACKgetri_");
      if (info>0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Zero-pivot in LU factorization");
      ierr = PetscLogFlops((PetscLogDouble)(4*n*n*n+3*n*n+5*n)/6);CHKERRQ(ierr); /* multiplications */
      ierr = PetscLogFlops((PetscLogDouble)(4*n*n*n-9*n*n+5*n)/6);CHKERRQ(ierr); /* additions */
      /* add values back into preconditioner matrix */
      ierr = MatSetValues(B,n,indices,n,indices,values,ADD_VALUES);CHKERRQ(ierr);
      ierr = PetscLogFlops((PetscLogDouble)(n*n));CHKERRQ(ierr);
    }
    ierr = IGAEndElement(iga,&element);CHKERRQ(ierr);

    ierr = ISLocalToGlobalMappingRestoreBlockIndices(map,&ltogmap);CHKERRQ(ierr);
    ierr = PetscFree2(indices,values);CHKERRQ(ierr);
    ierr = PetscFree(ipiv);CHKERRQ(ierr);
    ierr = PetscFree(work);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*
static PetscErrorCode PCSetFromOptions_EBE(PetscOptions *PetscOptionsObject,PC pc)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
*/

static PetscErrorCode PCApply_EBE(PC pc,Vec x,Vec y)
{
  PC_EBE         *ebe = (PC_EBE*)pc->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MatMult(ebe->mat,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTranspose_EBE(PC pc,Vec x,Vec y)
{
  PC_EBE         *ebe = (PC_EBE*)pc->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MatMultTranspose(ebe->mat,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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

static PetscErrorCode PCReset_EBE(PC pc)
{
  PC_EBE         *ebe = (PC_EBE*)pc->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MatDestroy(&ebe->mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_EBE(PC pc)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PCReset_EBE(pc);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


EXTERN_C_BEGIN
PetscErrorCode PCCreate_IGAEBE(PC pc);
PetscErrorCode PCCreate_IGAEBE(PC pc)
{
  PC_EBE         *ebe = NULL;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscNewLog(pc,&ebe);CHKERRQ(ierr);
  pc->data = (void*)ebe;

  pc->ops->setup               = PCSetUp_EBE;
  pc->ops->reset               = PCReset_EBE;
  pc->ops->destroy             = PCDestroy_EBE;
  pc->ops->setfromoptions      = NULL;/* PCSetFromOptions_EBE; */
  pc->ops->view                = PCView_EBE;
  pc->ops->apply               = PCApply_EBE;
  pc->ops->applytranspose      = PCApplyTranspose_EBE;
  PetscFunctionReturn(0);
}
EXTERN_C_END
