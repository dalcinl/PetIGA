#include "petiga.h"

#if PETSC_VERSION_(3,2,0)
#include "private/pcimpl.h"
#else
#include "petsc-private/pcimpl.h"
#endif

#include "petscblaslapack.h"
#if defined(PETSC_BLASLAPACK_UNDERSCORE)
   #include "petscblaslapack_uscore.h"
   #define sgetri_ sgetri_
   #define dgetri_ dgetri_
   #define qgetri_ qgetri_
   #define cgetri_ cgetri_
   #define zgetri_ zgetri_
#elif defined(PETSC_BLASLAPACK_CAPS)
   #define sgetri_ SGETRI
   #define dgetri_ DGETRI
   #define qgetri_ QGETRI
   #define cgetri_ CGETRI
   #define zgetri_ ZGETRI
#else /* (PETSC_BLASLAPACK_C) */
   #define sgetri_ sgetri
   #define dgetri_ dgetri
   #define qgetri_ qgetri
   #define cgetri_ cgetri
   #define zgetri_ zgetri
#endif
#if !defined(PETSC_USE_COMPLEX)
  #if defined(PETSC_USE_REAL_SINGLE)
    #define LAPACKgetri_ sgetri_
  #elif defined(PETSC_USE_REAL_DOUBLE)
    #define LAPACKgetri_ dgetri_
  #else /* (PETSC_USE_REAL_QUAD) */
    #define LAPACKgeqri_ qgeqri_
  #endif
#else
  #if defined(PETSC_USE_REAL_SINGLE)
    #define LAPACKgetri_ cgetri_
  #elif defined(PETSC_USE_REAL_DOUBLE)
    #define LAPACKgetri_ zgetri_
  #else /* (PETSC_USE_REAL_QUAD) */
    #error "LAPACKgetri_ not defined for quad complex"
  #endif
#endif
EXTERN_C_BEGIN
extern void LAPACKgetri_(PetscBLASInt*,PetscScalar*,PetscBLASInt*,
                         PetscBLASInt*,PetscScalar*,PetscBLASInt*,
                         PetscBLASInt*);
EXTERN_C_END


typedef struct {
  Mat mat;
} PC_EBE;

#undef  __FUNCT__
#define __FUNCT__ "PCSetFromOptions_EBE"
static PetscErrorCode PCSetFromOptions_EBE(PC pc)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

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
    Mat    Ad = 0;
    ierr = PetscTryMethod(A,"MatGetDiagonalBlock_C",(Mat,Mat*),(A,&Ad));CHKERRQ(ierr);
    if (Ad) {
      PetscBool baij,sbaij;
      PetscBool compressed,done;
      PetscInt  na,*ia,*ja;
      ierr = MatGetDiagonalBlock(A,&Ad);CHKERRQ(ierr);
      ierr = PetscTypeCompare((PetscObject)Ad,MATSEQBAIJ, &baij);CHKERRQ(ierr);
      ierr = PetscTypeCompare((PetscObject)Ad,MATSEQSBAIJ,&sbaij);CHKERRQ(ierr);
      compressed = (baij||sbaij) ? PETSC_TRUE: PETSC_FALSE;
      ierr = MatGetRowIJ(Ad,0,PETSC_FALSE,compressed,&na,&ia,&ja,&done);CHKERRQ(ierr);
      if (done) {
        PetscInt m,n,M,N,bs;
        PetscInt cstart,cend;
        PetscInt j,*newja;
        const MatType mtype;
        ierr = MatGetType(A,&mtype);
        ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
        ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
        ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);
        ierr = MatGetOwnershipRangeColumn(A,&cstart,&cend);CHKERRQ(ierr);

        ierr = MatCreate(comm,&mat);CHKERRQ(ierr);
        ierr = MatSetType(mat,mtype);CHKERRQ(ierr);
        ierr = MatSetSizes(mat,m,n,M,N);CHKERRQ(ierr);

        ierr = PetscMalloc1(ia[na],PetscInt,&newja);CHKERRQ(ierr);
        for (j=0; j<ia[na]; j++) newja[j] = ja[j] + cstart;
        ierr = MatMPIAIJSetPreallocationCSR  (mat,   ia,newja,PETSC_NULL);CHKERRQ(ierr);
        ierr = MatMPIBAIJSetPreallocationCSR (mat,bs,ia,newja,PETSC_NULL);CHKERRQ(ierr);
        ierr = MatMPISBAIJSetPreallocationCSR(mat,bs,ia,newja,PETSC_NULL);CHKERRQ(ierr);
        ierr = MatSetBlockSize(mat,bs);CHKERRQ(ierr);
        ierr = PetscFree(newja);CHKERRQ(ierr);
      }
      ierr = MatRestoreRowIJ(Ad,0,PETSC_FALSE,compressed,&na,&ia,&ja,&done);CHKERRQ(ierr);
    }
  }
  if (!mat) {
    MatDuplicateOption op = MAT_SHARE_NONZERO_PATTERN;
    #if PETSC_VERSION_(3,2,0)
    PetscBool sbaij = PETSC_FALSE;
    if (!sbaij) {ierr = PetscTypeCompare((PetscObject)A,MATSEQSBAIJ,&sbaij);CHKERRQ(ierr);}
    if (!sbaij) {ierr = PetscTypeCompare((PetscObject)A,MATMPISBAIJ,&sbaij);CHKERRQ(ierr);}
    ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);
    if (sbaij && bs==1) op = MAT_DO_NOT_COPY_VALUES;
    #endif
    ierr = MatDuplicate(A,op,&mat);CHKERRQ(ierr);
    ierr = MatSetOption(mat,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  }
  *B = mat;
  PetscFunctionReturn(0);
}

PetscInt ComputeOwnedGlobalIndices(const PetscInt lgmap[], PetscInt start, PetscInt end, PetscInt bs,
                                   PetscInt N, const PetscInt idx[], PetscInt idxout[])
{
  PetscInt i,j,Nout=0;
  for (i=0; i<N; i++) {
    PetscInt index = lgmap[idx[i]];
    if (index >= start && index < end)
      for (j=0; j<bs; j++)
        idxout[Nout++] = index*bs+j;
  }
  return Nout;
}

#undef  __FUNCT__
#define __FUNCT__ "PCSetUp_EBE"
static PetscErrorCode PCSetUp_EBE(PC pc)
{
  PC_EBE         *ebe = (PC_EBE*)pc->data;
  Mat            A,B;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  A = pc->pmat;
  if (pc->flag != SAME_NONZERO_PATTERN) {
    ierr = MatDestroy(&ebe->mat);CHKERRQ(ierr);
  }
  if (!ebe->mat) {
    ierr = PCSetUp_EBE_CreateMatrix(A,&ebe->mat);CHKERRQ(ierr);
  }
  B = ebe->mat;

  {
    IGA          iga = 0;
    IGAElement   element;
    PetscInt     nen,dof;
    PetscInt     n,*idx;
    PetscScalar  *vals,*work,lwkopt;
    PetscBLASInt m,*ipiv,info,lwork;
    PetscInt     start,end;
    const PetscInt *lgmap;
    ISLocalToGlobalMapping map;

    ierr = PetscObjectQuery((PetscObject)A,"IGA",(PetscObject*)&iga);CHKERRQ(ierr);
    if (!iga) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Matrix is missing the IGA context");
    PetscValidHeaderSpecific(iga,IGA_CLASSID,1);

    ierr = IGAGetElement(iga,&element);CHKERRQ(ierr);
    ierr = IGAElementGetInfo(element,0,&nen,&dof,0);CHKERRQ(ierr);

    if (dof == 1) {
      ierr = MatGetLocalToGlobalMapping(A,&map,PETSC_NULL);CHKERRQ(ierr);
    } else {
      ierr = MatGetLocalToGlobalMappingBlock(A,&map,PETSC_NULL);CHKERRQ(ierr);
    }
    ierr = ISLocalToGlobalMappingGetIndices(map,&lgmap);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(A,&start,&end);CHKERRQ(ierr);
    start /= dof; end /= dof;

    n = nen*dof;
    ierr = PetscMalloc1(  n,PetscInt,   &idx);CHKERRQ(ierr);
    ierr = PetscMalloc1(n*n,PetscScalar,&vals);CHKERRQ(ierr);
    m = PetscBLASIntCast(n); lwork = -1; work = &lwkopt;
    ierr = PetscMalloc1(m,PetscBLASInt,&ipiv);CHKERRQ(ierr);
    LAPACKgetri_(&m,vals,&m,ipiv,work,&lwork,&info);
    lwork = (info==0) ? (PetscBLASInt)work[0] : m*128;
    ierr = PetscMalloc1(lwork,PetscScalar,&work);CHKERRQ(ierr);

    ierr = MatZeroEntries(B);CHKERRQ(ierr);
    ierr = IGAElementBegin(element);CHKERRQ(ierr);
    while (IGAElementNext(element)) {
      const PetscInt *indices = element->mapping;
      m = n = ComputeOwnedGlobalIndices(lgmap,start,end,dof,nen,indices,idx);
      /* get element matrix from global matrix */
      ierr = MatGetValues(A,n,idx,n,idx,vals);;CHKERRQ(ierr);
      /* compute inverse of element matrix */
      LAPACKgetrf_(&m,&m,vals,&m,ipiv,&info);
      if (info<0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Bad argument to LAPACKgetrf_");
      if (info>0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Zero-pivot in LU factorization");
      ierr = PetscLogFlops((1/3.*n*n*n         +2/3.*n));CHKERRQ(ierr); /* multiplications */
      ierr = PetscLogFlops((1/3.*n*n*n-1/2.*n*n+1/6.*n));CHKERRQ(ierr); /* additions */
      LAPACKgetri_(&m,vals,&m,ipiv,work,&lwork,&info);
      if (info<0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Bad argument to LAPACKgetri_");
      if (info>0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Zero-pivot in LU factorization");
      ierr = PetscLogFlops((2/3.*n*n*n+1/2.*n*n+5/6.*n));CHKERRQ(ierr); /* multiplications */
      ierr = PetscLogFlops((2/3.*n*n*n-3/2.*n*n+5/6.*n));CHKERRQ(ierr); /* additions */
      /* add values back into preconditioner matrix */
      ierr = MatSetValues(B,n,idx,n,idx,vals,ADD_VALUES);CHKERRQ(ierr);
      ierr = PetscLogFlops(n*n);CHKERRQ(ierr);
    }
    ierr = IGAElementEnd(element);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd  (B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    ierr = ISLocalToGlobalMappingRestoreIndices(map,&lgmap);CHKERRQ(ierr);
    ierr = PetscFree(idx);CHKERRQ(ierr);
    ierr = PetscFree(vals);CHKERRQ(ierr);
    ierr = PetscFree(ipiv);CHKERRQ(ierr);
    ierr = PetscFree(work);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

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
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (!isascii) PetscFunctionReturn(0);
  ierr = PetscViewerASCIIPrintf(viewer,"element-by-element preconditioner matrix:\n");CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
  ierr = MatView(ebe->mat,viewer);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
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
#define __FUNCT__ "PCCreate_EBE"
PetscErrorCode PCCreate_EBE(PC pc)
{
  PC_EBE         *ebe = 0;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscNewLog(pc,PC_EBE,&ebe);CHKERRQ(ierr);
  pc->data = (void*)ebe;

  pc->ops->setup               = PCSetUp_EBE;
  pc->ops->reset               = PCReset_EBE;
  pc->ops->destroy             = PCDestroy_EBE;
  pc->ops->setfromoptions      = PCSetFromOptions_EBE;
  pc->ops->view                = PCView_EBE;
  pc->ops->apply               = PCApply_EBE;
  pc->ops->applytranspose      = PCApplyTranspose_EBE;

  PetscFunctionReturn(0);
}
EXTERN_C_END
