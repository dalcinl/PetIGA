#include "petiga.h"

PetscErrorCode IGAOptionsAlias(const char alias[],const char defval[],const char name[])
{
  const char     *prefix = NULL; /* XXX */
  char           value[1024]= {0};
  PetscBool      flag = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(alias,1);
  PetscValidCharPointer(name,3);
  ierr = PetscOptionsHasName(NULL,NULL,alias,&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = PetscOptionsGetString(NULL,NULL,alias,value,sizeof(value),&flag);CHKERRQ(ierr);
  } else if (defval) {
    ierr = PetscOptionsHasName(NULL,prefix,name,&flag);CHKERRQ(ierr);
    if (flag) PetscFunctionReturn(0);
    ierr = PetscStrncpy(value,defval,sizeof(value));CHKERRQ(ierr);
  } else PetscFunctionReturn(0);
  if (prefix && prefix[0]) {ierr = PetscOptionsPrefixPush(NULL,prefix);CHKERRQ(ierr);}
  ierr = PetscOptionsSetValue(NULL,name,value);CHKERRQ(ierr);
  if (prefix && prefix[0]) {ierr = PetscOptionsPrefixPop(NULL);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode IGAOptionsDefault(const char prefix[],const char name[],const char value[])
{
  PetscBool      flag = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(NULL,prefix,name,&flag);CHKERRQ(ierr);
  if (flag) PetscFunctionReturn(0);
  if (prefix && prefix[0]) {ierr = PetscOptionsPrefixPush(NULL,prefix);CHKERRQ(ierr);}
  ierr = PetscOptionsSetValue(NULL,name,value);CHKERRQ(ierr);
  if (prefix && prefix[0]) {ierr = PetscOptionsPrefixPop(NULL);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode IGAOptionsReject(const char prefix[],const char name[])
{
  PetscBool      flag = PETSC_FALSE;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscOptionsHasName(NULL,prefix,name,&flag);CHKERRQ(ierr);
  if (flag) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Disabled option: %s",name);
  PetscFunctionReturn(0);
}

PetscEnum IGAGetOptEnum(const char prefix[],const char name[],const char *const elist[],PetscEnum defval)
{
  PetscErrorCode ierr;
  ierr = PetscOptionsGetEnum(NULL,prefix,name,elist,&defval,NULL);CHKERRABORT(PETSC_COMM_WORLD,ierr);
  return defval;
}

const char* IGAGetOptString(const char prefix[],const char name[],const char defval[])
{
  PetscErrorCode ierr; static char buffer[1024];
  ierr = PetscStrncpy(buffer,defval,sizeof(buffer));CHKERRABORT(PETSC_COMM_WORLD,ierr);
  ierr = PetscOptionsGetString(NULL,prefix,name,buffer,sizeof(buffer),NULL);CHKERRABORT(PETSC_COMM_WORLD,ierr);
  return buffer;
}

PetscBool IGAGetOptBool(const char prefix[],const char name[],PetscBool defval)
{
  PetscErrorCode ierr;
  ierr = PetscOptionsGetBool(NULL,prefix,name,&defval,NULL);CHKERRABORT(PETSC_COMM_WORLD,ierr);
  return defval;
}

PetscInt IGAGetOptInt(const char prefix[],const char name[],PetscInt defval)
{
  PetscErrorCode ierr;
  ierr = PetscOptionsGetInt(NULL,prefix,name,&defval,NULL);CHKERRABORT(PETSC_COMM_WORLD,ierr);
  return defval;
}

PetscReal IGAGetOptReal(const char prefix[],const char name[],PetscReal defval)
{
  PetscErrorCode ierr;
  ierr = PetscOptionsGetReal(NULL,prefix,name,&defval,NULL);CHKERRABORT(PETSC_COMM_WORLD,ierr);
  return defval;
}

PetscScalar IGAGetOptScalar(const char prefix[],const char name[],PetscScalar defval)
{
  PetscErrorCode ierr;
  ierr = PetscOptionsGetScalar(NULL,prefix,name,&defval,NULL);CHKERRABORT(PETSC_COMM_WORLD,ierr);
  return defval;
}
