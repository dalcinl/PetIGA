#include "petiga.h"
#include <petsc/private/pcimpl.h>
#include <petsc/private/tsimpl.h>
#include <petsc/private/dmimpl.h>

PETSC_EXTERN PetscBool IGAPackageInitialized;
PETSC_EXTERN PetscBool IGARegisterAllCalled;

PETSC_EXTERN PetscFunctionList PCList;
PETSC_EXTERN PetscFunctionList TSList;
PETSC_EXTERN PetscFunctionList DMList;

EXTERN_C_BEGIN
extern PetscErrorCode PCCreate_IGAEBE(PC);
extern PetscErrorCode PCCreate_IGABBB(PC);
EXTERN_C_END

EXTERN_C_BEGIN
extern PetscErrorCode DMCreate_IGA(DM);
EXTERN_C_END

PetscBool IGAPackageInitialized = PETSC_FALSE;
PetscBool IGARegisterAllCalled  = PETSC_FALSE;

PetscClassId  IGA_CLASSID = 0;

PetscLogEvent IGA_FormScalar = 0;
PetscLogEvent IGA_FormVector = 0;
PetscLogEvent IGA_FormMatrix = 0;
PetscLogEvent IGA_FormSystem = 0;
PetscLogEvent IGA_FormFunction = 0;
PetscLogEvent IGA_FormJacobian = 0;
PetscLogEvent IGA_FormIFunction = 0;
PetscLogEvent IGA_FormIJacobian = 0;

PetscErrorCode IGARegisterAll(void)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  IGARegisterAllCalled = PETSC_TRUE;
  ierr = PCRegisterAll();CHKERRQ(ierr);
  ierr = PCRegister(PCIGAEBE,PCCreate_IGAEBE);CHKERRQ(ierr);
  ierr = PCRegister(PCIGABBB,PCCreate_IGABBB);CHKERRQ(ierr);
  ierr = DMRegisterAll();CHKERRQ(ierr);
  ierr = DMRegister(DMIGA,DMCreate_IGA);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IGAFinalizePackage(void)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (PCList) {ierr = PetscFunctionListDestroy(&PCList);CHKERRQ(ierr);}
  if (TSList) {ierr = PetscFunctionListDestroy(&TSList);CHKERRQ(ierr);}
  if (DMList) {ierr = PetscFunctionListDestroy(&DMList);CHKERRQ(ierr);}
  IGAPackageInitialized = PETSC_FALSE;
  IGARegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#if PETSC_VERSION_LT(3,9,0)
static PetscErrorCode PetscStrInList(const char str[],const char list[],char sep,PetscBool *found)
{
  PetscToken     token;
  char           *item;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *found = PETSC_FALSE;
  ierr = PetscTokenCreate(list,sep,&token);CHKERRQ(ierr);
  ierr = PetscTokenFind(token,&item);CHKERRQ(ierr);
  while (item) {
    ierr = PetscStrcmp(str,item,found);CHKERRQ(ierr);
    if (*found) break;
    ierr = PetscTokenFind(token,&item);CHKERRQ(ierr);
  }
  ierr = PetscTokenDestroy(&token);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

PetscErrorCode IGAInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;
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
  ierr = PetscLogEventRegister("IGAFormVector",IGA_CLASSID,&IGA_FormVector);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("IGAFormMatrix",IGA_CLASSID,&IGA_FormMatrix);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("IGAFormSystem",IGA_CLASSID,&IGA_FormSystem);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("IGAFormFunction",IGA_CLASSID,&IGA_FormFunction);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("IGAFormJacobian",IGA_CLASSID,&IGA_FormJacobian);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("IGAFormIFunction",IGA_CLASSID,&IGA_FormIFunction);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("IGAFormIJacobian",IGA_CLASSID,&IGA_FormIJacobian);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-info_exclude",logList,sizeof(logList),&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrInList("iga",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {ierr = PetscInfoDeactivateClass(IGA_CLASSID);CHKERRQ(ierr);}
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrInList("iga",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {ierr = PetscLogEventDeactivateClass(IGA_CLASSID);CHKERRQ(ierr);}
  }
  /* Register package finalizer */
  ierr = PetscRegisterFinalize(IGAFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
EXTERN_C_BEGIN
PetscErrorCode PetscDLLibraryRegister_petiga(void);
PetscErrorCode PetscDLLibraryRegister_petiga(void)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = IGAInitializePackage();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
#endif
