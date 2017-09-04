#include <petiga.h>

typedef struct {
  PetscReal lambda;
} Params;

EXTERN_C_BEGIN
extern PetscErrorCode FunctionC99(IGAPoint,const PetscScalar U[],PetscScalar F[],void *ctx);
extern PetscErrorCode JacobianC99(IGAPoint,const PetscScalar U[],PetscScalar J[],void *ctx);
EXTERN_C_END

EXTERN_C_BEGIN
extern PetscErrorCode FunctionF90(IGAPoint,const PetscScalar U[],PetscScalar F[],void *ctx);
extern PetscErrorCode JacobianF90(IGAPoint,const PetscScalar U[],PetscScalar J[],void *ctx);
EXTERN_C_END

EXTERN_C_BEGIN
extern PetscErrorCode FunctionCXX(IGAPoint,const PetscScalar U[],PetscScalar F[],void *ctx);
extern PetscErrorCode JacobianCXX(IGAPoint,const PetscScalar U[],PetscScalar J[],void *ctx);
EXTERN_C_END

EXTERN_C_BEGIN
extern PetscErrorCode FunctionFAD(IGAPoint,const PetscScalar U[],PetscScalar F[],void *ctx);
extern PetscErrorCode JacobianFAD(IGAPoint,const PetscScalar U[],PetscScalar J[],void *ctx);
EXTERN_C_END

int main(int argc, char *argv[]) {
  PetscInitialize(&argc,&argv,NULL,NULL);

  IGAOptionsAlias("-dim",  "2", "-iga_dim");
  IGAOptionsAlias("-deg",  "2", "-iga_degree");
  IGAOptionsAlias("-nel", "64", "-iga_elements");

  IGA iga;
  IGACreate(PETSC_COMM_WORLD,&iga);
  IGASetDof(iga,1);
  IGASetOrder(iga,1);
  IGASetFromOptions(iga);
  IGASetUp(iga);

  IGASetFieldName(iga,0,"u");

  int dim;
  IGAGetDim(iga,&dim);
  int direction,side;
  for (direction=0; direction<dim; direction++) {
    for (side=0; side<2; side++) {
      int field = 0; PetscScalar value = 0.0;
      IGASetBoundaryValue(iga,direction,side,field,value);
    }
  }

  Params params;
  params.lambda = IGAGetOptReal(NULL,"-lambda",6.80);

  PetscBool c99 = IGAGetOptBool(NULL,"-c99",PETSC_TRUE);
  if (c99) {
    IGASetFormFunction(iga,FunctionC99,&params);
    IGASetFormJacobian(iga,JacobianC99,&params);
  }
  PetscBool f90 = IGAGetOptBool(NULL,"-f90",PETSC_FALSE);
  if (f90) {
    IGASetFormFunction(iga,FunctionF90,&params);
    IGASetFormJacobian(iga,JacobianF90,&params);
  }
  PetscBool cxx = IGAGetOptBool(NULL,"-cxx",PETSC_FALSE);
  if (cxx) {
    IGASetFormFunction(iga,FunctionCXX,&params);
    IGASetFormJacobian(iga,JacobianCXX,&params);
  }
  PetscBool fad = IGAGetOptBool(NULL,"-fad",PETSC_FALSE);
  if (fad) {
    IGASetFormFunction(iga,FunctionFAD,&params);
    IGASetFormJacobian(iga,JacobianFAD,&params);
  }
  PetscBool fd = IGAGetOptBool(NULL,"-fd",PETSC_FALSE);
  if (fd) {IGASetFormJacobian(iga,IGAFormJacobianFD,&params);}

  SNES snes;
  IGACreateSNES(iga,&snes);
  SNESSetFromOptions(snes);

  Vec U;
  IGACreateVec(iga,&U);
  SNESSolve(snes,NULL,U);

  VecViewFromOptions(U,NULL,"-output");

  VecDestroy(&U);
  SNESDestroy(&snes);
  IGADestroy(&iga);
  PetscFinalize();
  return 0;
}
