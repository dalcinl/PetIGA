#include "petiga.h"

PETSC_EXTERN PetscBool     IGARegisterAllCalled;

#if PETSC_VERSION_LE(3,3,0)
#define PCRegisterAll() PCRegisterAll(0)
#define TSRegisterAll() TSRegisterAll(0)
#define PCRegister(s,f) PCRegister(s,0,0,f)
#define TSRegister(s,f) TSRegister(s,0,0,f)
#define PetscFunctionList        PetscFList
#define PetscFunctionListDestroy PetscFListDestroy
#endif

PETSC_EXTERN PetscFunctionList PCList;
PETSC_EXTERN PetscFunctionList TSList;

EXTERN_C_BEGIN
extern PetscErrorCode PCCreate_IGAEBE(PC);
extern PetscErrorCode PCCreate_IGABBB(PC);
EXTERN_C_END

EXTERN_C_BEGIN
extern PetscErrorCode TSCreate_Alpha2(TS);
EXTERN_C_END

EXTERN_C_BEGIN
extern PetscErrorCode SNESSetFromOptions_FDColoring(SNES);
EXTERN_C_END

static PetscBool IGAPackageInitialized = PETSC_FALSE;
PetscBool        IGARegisterAllCalled  = PETSC_FALSE;

PetscClassId  IGA_CLASSID = 0;

PetscLogEvent IGA_FormScalar = 0;
PetscLogEvent IGA_FormSystem = 0;
PetscLogEvent IGA_FormFunction = 0;
PetscLogEvent IGA_FormJacobian = 0;
PetscLogEvent IGA_FormIFunction = 0;
PetscLogEvent IGA_FormIJacobian = 0;

#undef  __FUNCT__
#define __FUNCT__ "IGARegisterAll"
PetscErrorCode IGARegisterAll(void)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  IGARegisterAllCalled = PETSC_TRUE;
  ierr = PCRegisterAll();CHKERRQ(ierr);
  ierr = PCRegister(PCIGAEBE,PCCreate_IGAEBE);CHKERRQ(ierr);
  ierr = PCRegister(PCIGABBB,PCCreate_IGABBB);CHKERRQ(ierr);
  ierr = TSRegisterAll();CHKERRQ(ierr);
  ierr = TSRegister(TSALPHA2,TSCreate_Alpha2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAFinalizePackage"
PetscErrorCode IGAFinalizePackage(void)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (PCList) {ierr = PetscFunctionListDestroy(&PCList);CHKERRQ(ierr);}
  if (TSList) {ierr = PetscFunctionListDestroy(&TSList);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAInitializePackage"
PetscErrorCode IGAInitializePackage(void)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (IGAPackageInitialized) PetscFunctionReturn(0);
  IGAPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("IGA",&IGA_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = IGARegisterAll();CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("IGAFormScalar",IGA_CLASSID,&IGA_FormScalar);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("IGAFormSystem",IGA_CLASSID,&IGA_FormSystem);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("IGAFormFunction",IGA_CLASSID,&IGA_FormFunction);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("IGAFormJacobian",IGA_CLASSID,&IGA_FormJacobian);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("IGAFormIFunction",IGA_CLASSID,&IGA_FormIFunction);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("IGAFormIJacobian",IGA_CLASSID,&IGA_FormIJacobian);CHKERRQ(ierr);
  /* Additional option handlers */
  ierr = SNESAddOptionsChecker(SNESSetFromOptions_FDColoring);CHKERRQ(ierr);
  /* Register finalization routine */
  ierr = PetscRegisterFinalize(IGAFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#ifdef PETSC_USE_DYNAMIC_LIBRARIES
EXTERN_C_BEGIN
#undef  __FUNCT__
#define __FUNCT__ "PetscDLLibraryRegister_petiga"
#if PETSC_VERSION_LE(3,3,3)
PetscErrorCode PetscDLLibraryRegister_petiga(const char path[])
#else
PetscErrorCode PetscDLLibraryRegister_petiga(void)
#endif
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = IGAInitializePackage();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
#endif
