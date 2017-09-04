#include "petiga.h"
#include "petigagrid.h"
#include <petscblaslapack.h>
#include <petsc/private/pcimpl.h>

typedef struct {
  PetscInt dim,dof;
  PetscInt overlap[3];
  PetscInt ghost_start[3];
  PetscInt ghost_width[3];
  LGMap    lgmap;
  Mat      mat;
} PC_BBB;

PETSC_STATIC_INLINE
PetscInt Index3D(const PetscInt start[3],const PetscInt width[3],
                 PetscInt i,PetscInt j,PetscInt k)
{
  if (start) { i -= start[0]; j -= start[1]; k -= start[2]; }
  return i + j * width[0] + k * width[0] * width[1];
}

static PetscInt ComputeOverlap(const PetscInt lgmap[],PetscInt bs,
                               PetscInt Astart,PetscInt Aend,
                               const PetscInt gstart[3],const PetscInt gwidth[3],
                               const PetscInt overlap[3],
                               PetscInt iA,PetscInt jA,PetscInt kA,
                               PetscInt indices[])
{
  PetscInt igs = gstart[0], ige = gstart[0]+gwidth[0], iov = overlap[0];
  PetscInt jgs = gstart[1], jge = gstart[1]+gwidth[1], jov = overlap[1];
  PetscInt kgs = gstart[2], kge = gstart[2]+gwidth[2], kov = overlap[2];
  PetscInt i, iL = PetscMax(iA-iov,igs), iR = PetscMin(iA+iov,ige-1);
  PetscInt j, jL = PetscMax(jA-jov,jgs), jR = PetscMin(jA+jov,jge-1);
  PetscInt k, kL = PetscMax(kA-kov,kgs), kR = PetscMin(kA+kov,kge-1);
  PetscInt c, pos = 0;
  for (i=iL; i<=iR; i++)
    for (j=jL; j<=jR; j++)
      for (k=kL; k<=kR; k++) {
        PetscInt Alocal = Index3D(gstart,gwidth,i,j,k);
        PetscInt Aglobal = lgmap[Alocal];
        if (PetscUnlikely(Aglobal < Astart || Aglobal >= Aend)) continue;
        for (c=0; c<bs; c++) indices[pos++] = c + bs*Aglobal;
      }
  return pos;
}

PETSC_STATIC_INLINE
PetscErrorCode InferMatrixType(Mat A,PetscBool *aij,PetscBool *baij,PetscBool *sbaij)
{
  void (*f)(void) = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *aij = *baij = *sbaij = PETSC_FALSE;
  if (!f) {ierr = PetscObjectQueryFunction((PetscObject)A,"MatMPIAIJSetPreallocation_C",&f);CHKERRQ(ierr);}
  if (!f) {ierr = PetscObjectQueryFunction((PetscObject)A,"MatSeqAIJSetPreallocation_C",&f);CHKERRQ(ierr);}
  if  (f) {*aij = PETSC_TRUE; goto done;};
  if (!f) {ierr = PetscObjectQueryFunction((PetscObject)A,"MatMPIBAIJSetPreallocation_C",&f);CHKERRQ(ierr);}
  if (!f) {ierr = PetscObjectQueryFunction((PetscObject)A,"MatSeqBAIJSetPreallocation_C",&f);CHKERRQ(ierr);}
  if  (f) {*baij = PETSC_TRUE; goto done;};
  if (!f) {ierr = PetscObjectQueryFunction((PetscObject)A,"MatMPISBAIJSetPreallocation_C",&f);CHKERRQ(ierr);}
  if (!f) {ierr = PetscObjectQueryFunction((PetscObject)A,"MatSeqSBAIJSetPreallocation_C",&f);CHKERRQ(ierr);}
  if  (f) {*sbaij = PETSC_TRUE; goto done;};
 done:
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_BBB_CreateMatrix(PC_BBB *bbb,Mat A,Mat *B)
{
  MPI_Comm       comm = ((PetscObject)A)->comm;
  PetscBool      aij,baij,sbaij;
  PetscInt       m,n,M,N,bs;
  MatType        mtype;
  Mat            mat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MatGetType(A,&mtype);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);

  ierr = MatCreate(comm,&mat);CHKERRQ(ierr);
  ierr = MatSetSizes(mat,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetBlockSize(mat,bs);CHKERRQ(ierr);
  ierr = MatSetType(mat,mtype);CHKERRQ(ierr);
  ierr = InferMatrixType(mat,&aij,&baij,&sbaij);CHKERRQ(ierr);

 if (aij || baij || sbaij) {
   PetscInt i, dim = bbb->dim, dof = bbb->dof;
   PetscInt *overlap = bbb->overlap;
   PetscInt dnnz=1, onnz=0;
   for (i=0; i<dim; i++) dnnz *= (4*overlap[i] + 1);
   if (aij) {
     dnnz *= dof; onnz *= dof;
     ierr = MatSeqAIJSetPreallocation(mat,dnnz,0);CHKERRQ(ierr);
     ierr = MatMPIAIJSetPreallocation(mat,dnnz,0,onnz,0);CHKERRQ(ierr);
   } else if (baij) {
     ierr = MatSeqBAIJSetPreallocation(mat,dof,dnnz,0);CHKERRQ(ierr);
     ierr = MatMPIBAIJSetPreallocation(mat,dof,dnnz,0,onnz,0);CHKERRQ(ierr);
   } else if (sbaij) {
     ierr = MatSeqSBAIJSetPreallocation(mat,dof,dnnz,0);CHKERRQ(ierr);
     ierr = MatMPISBAIJSetPreallocation(mat,dof,dnnz,0,onnz,0);CHKERRQ(ierr);
   }
   ierr = MatSetOption(mat,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
 } else {
   ierr = MatSetUp(mat);CHKERRQ(ierr);
 }
 *B = mat;
 PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_BBB(PC pc)
{
  PC_BBB         *bbb = (PC_BBB*)pc->data;
  IGA            iga = 0;
  Mat            A,B;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  A = pc->pmat;
  ierr = PetscObjectQuery((PetscObject)A,"IGA",(PetscObject*)&iga);CHKERRQ(ierr);
  if (!iga) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Matrix is missing the IGA context");
  PetscValidHeaderSpecific(iga,IGA_CLASSID,0);

  if (!pc->setupcalled) {
    MPI_Comm comm;
    IGA_Grid grid;
    PetscInt i, dim = iga->dim;
    const PetscInt *sizes = iga->node_sizes;
    const PetscInt *lstart = iga->node_lstart;
    const PetscInt *lwidth = iga->node_lwidth;
    PetscInt *overlap = bbb->overlap;
    PetscInt *gstart  = bbb->ghost_start;
    PetscInt *gwidth  = bbb->ghost_width;
    bbb->dim = iga->dim;
    bbb->dof = iga->dof;
    for (i=0; i<dim; i++) {
      PetscInt p = iga->axis[i]->p;
      if (overlap[i] < 0) {
        overlap[i] = p/2;
      } else {
        overlap[i] = PetscMin(overlap[i], p);
      }
    }
    for (i=0; i<dim; i++) {
      gstart[i] = lstart[i] - overlap[i];
      gwidth[i] = lwidth[i] + overlap[i];
      if (gstart[i] < 0)
        gstart[i] = iga->node_gstart[i];
      if (gstart[i]+gwidth[i] >= sizes[i])
        gwidth[i] = iga->node_gwidth[i];
    }
    for (i=dim; i<3; i++) {
      overlap[i] = 0;
      gstart[i]  = 0;
      gwidth[i]  = 1;
    }
    ierr = IGAGetComm(iga,&comm);CHKERRQ(ierr);
    ierr = IGA_Grid_Create(comm,&grid);CHKERRQ(ierr);
    ierr = IGA_Grid_Init(grid,iga->dim,1,sizes,lstart,lwidth,gstart,gwidth);CHKERRQ(ierr);
    ierr = IGA_Grid_SetAO(grid,iga->ao);CHKERRQ(ierr);
    ierr = IGA_Grid_GetLGMap(grid,&bbb->lgmap);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)bbb->lgmap);CHKERRQ(ierr);
    ierr = IGA_Grid_Destroy(&grid);CHKERRQ(ierr);
  }

  if (pc->flag != SAME_NONZERO_PATTERN) {
    ierr = MatDestroy(&bbb->mat);CHKERRQ(ierr);
  }
  if (!bbb->mat) {
    ierr = PCSetUp_BBB_CreateMatrix(bbb,A,&bbb->mat);CHKERRQ(ierr);
  }
  B = bbb->mat;

  ierr = MatZeroEntries(B);CHKERRQ(ierr);
  {
    PetscInt       i,j,k,dim,dof;
    const PetscInt *start,*width;
    const PetscInt *gstart,*gwidth;
    const PetscInt *overlap;
    const PetscInt *ltogmap;
    PetscInt       rstart,rend;
    PetscInt       n,*indices;
    PetscBLASInt   m,*ipiv,info,lwork;
    PetscScalar    *values,*work,lwkopt;

    ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
    ierr = IGAGetDof(iga,&dof);CHKERRQ(ierr);

    start   = iga->node_lstart;
    width   = iga->node_lwidth;

    gstart  = bbb->ghost_start;
    gwidth  = bbb->ghost_width;
    overlap = bbb->overlap;
    ierr = ISLocalToGlobalMappingGetBlockIndices(bbb->lgmap,&ltogmap);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
    rstart /= dof; rend /= dof;

    for (n=dof, i=0; i<dim; n *= (2*overlap[i++] + 1));

    ierr = PetscBLASIntCast(n,&m);CHKERRQ(ierr);
    ierr = PetscMalloc1((size_t)n,&indices);CHKERRQ(ierr);
    ierr = PetscMalloc1((size_t)n*(size_t)n,&values);CHKERRQ(ierr);
    ierr = PetscMalloc1((size_t)m,&ipiv);CHKERRQ(ierr);
    lwork = -1; work = &lwkopt;
    LAPACKgetri_(&m,values,&m,ipiv,work,&lwork,&info);
    lwork = (info==0) ? (PetscBLASInt)PetscRealPart(work[0]) : m*128;
    ierr = PetscMalloc1((size_t)lwork,&work);CHKERRQ(ierr);

    for (k=start[2]; k<start[2]+width[2]; k++)
      for (j=start[1]; j<start[1]+width[1]; j++)
        for (i=start[0]; i<start[0]+width[0]; i++)
          {
            n = ComputeOverlap(ltogmap,dof,rstart,rend,gstart,gwidth,overlap,i,j,k,indices);CHKERRQ(ierr);
            /* get element matrix from global matrix */
            ierr = MatGetValues(A,n,indices,n,indices,values);CHKERRQ(ierr);
            /* compute inverse of element matrix */
            if (PetscLikely(n > 1)) {
              ierr = PetscBLASIntCast(n,&m);CHKERRQ(ierr);
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
            } else if (PetscLikely(n == 1)) {
              if (PetscAbsScalar(values[0]) >  0) values[0] = (PetscScalar)1.0/values[0];
              ierr = PetscLogFlops(1);CHKERRQ(ierr);
            }
            /* add values back into preconditioner matrix */
            ierr = MatSetValues(B,n,indices,n,indices,values,ADD_VALUES);CHKERRQ(ierr);
            ierr = PetscLogFlops((PetscLogDouble)(n*n));CHKERRQ(ierr);
          }

    ierr = ISLocalToGlobalMappingRestoreBlockIndices(bbb->lgmap,&ltogmap);CHKERRQ(ierr);
    ierr = PetscFree2(indices,values);CHKERRQ(ierr);
    ierr = PetscFree(ipiv);CHKERRQ(ierr);
    ierr = PetscFree(work);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_BBB(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_BBB         *bbb = (PC_BBB*)pc->data;
  PetscBool      flg;
  PetscInt       i,no=3,overlap[3];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (i=0; i<3; i++) overlap[i] = bbb->overlap[i];
  ierr = PetscOptionsIntArray("-pc_bbb_overlap","Overlap","",overlap,&no,&flg);CHKERRQ(ierr);
  if (flg) for (i=0; i<3; i++) {
      PetscInt ov = (i<no) ? overlap[i] : overlap[0];
      bbb->overlap[i] = ov;
    }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_BBB(PC pc,Vec x,Vec y)
{
  PC_BBB         *bbb = (PC_BBB*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMult(bbb->mat,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTranspose_BBB(PC pc,Vec x,Vec y)
{
  PC_BBB         *bbb = (PC_BBB*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultTranspose(bbb->mat,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_BBB(PC pc,PetscViewer viewer)
{
  PC_BBB         *bbb = (PC_BBB*)pc->data;
  PetscInt       *ov = bbb->overlap;
  PetscBool      isascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (!isascii) PetscFunctionReturn(0);
  if (!bbb->mat) PetscFunctionReturn(0);
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"overlap: %D,%D,%D\n",ov[0],ov[1],ov[2]);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"basis-by-basis matrix:\n");CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
  ierr = MatView(bbb->mat,viewer);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCReset_BBB(PC pc)
{
  PC_BBB         *bbb = (PC_BBB*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISLocalToGlobalMappingDestroy(&bbb->lgmap);CHKERRQ(ierr);
  ierr = MatDestroy(&bbb->mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_BBB(PC pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCReset_BBB(pc);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
PetscErrorCode PCCreate_IGABBB(PC pc);
PetscErrorCode PCCreate_IGABBB(PC pc)
{
  PC_BBB         *bbb = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(pc,&bbb);CHKERRQ(ierr);
  pc->data = (void*)bbb;

  bbb->overlap[0] = PETSC_DECIDE;
  bbb->overlap[1] = PETSC_DECIDE;
  bbb->overlap[2] = PETSC_DECIDE;
  bbb->mat        = NULL;

  pc->ops->setup               = PCSetUp_BBB;
  pc->ops->reset               = PCReset_BBB;
  pc->ops->destroy             = PCDestroy_BBB;
  pc->ops->setfromoptions      = PCSetFromOptions_BBB;
  pc->ops->view                = PCView_BBB;
  pc->ops->apply               = PCApply_BBB;
  pc->ops->applytranspose      = PCApplyTranspose_BBB;
  PetscFunctionReturn(0);
}
EXTERN_C_END
