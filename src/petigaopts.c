#include "petiga.h"

#undef  __FUNCT__
#define __FUNCT__ "IGAOptionsAlias"
PetscErrorCode IGAOptionsAlias(const char alias[],const char defval[],const char name[])
{
  const char     *prefix = NULL; /* XXX */
  char           value[1024]= {0};
  PetscBool      flag = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(alias,1);
  PetscValidCharPointer(name,3);
  ierr = PetscOptionsHasName(NULL,alias,&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = PetscOptionsGetString(NULL,alias,value,sizeof(value),&flag);CHKERRQ(ierr);
  } else if (defval) {
    ierr = PetscOptionsHasName(prefix,name,&flag);CHKERRQ(ierr);
    if (flag) PetscFunctionReturn(0);
    ierr = PetscStrncpy(value,defval,sizeof(value));CHKERRQ(ierr);
  }
  if (prefix && prefix[0]) {ierr = PetscOptionsPrefixPush(prefix);CHKERRQ(ierr);}
  ierr = PetscOptionsSetValue(name,value);CHKERRQ(ierr);
  if (prefix && prefix[0]) {ierr = PetscOptionsPrefixPop();CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAOptionsDefault"
PetscErrorCode IGAOptionsDefault(const char prefix[],const char name[],const char value[])
{
  PetscBool      flag = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(prefix,name,&flag);CHKERRQ(ierr);
  if (flag) PetscFunctionReturn(0);
  if (prefix && prefix[0]) {ierr = PetscOptionsPrefixPush(prefix);CHKERRQ(ierr);}
  ierr = PetscOptionsSetValue(name,value);CHKERRQ(ierr);
  if (prefix && prefix[0]) {ierr = PetscOptionsPrefixPop();CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAOptionsReject"
PetscErrorCode IGAOptionsReject(const char prefix[],const char name[])
{
  PetscBool      flag = PETSC_FALSE;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscOptionsHasName(prefix,name,&flag);CHKERRQ(ierr);
  if (flag) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Disabled option: %s",name);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAGetOptEnum"
PetscEnum IGAGetOptEnum(const char prefix[],const char name[],const char const *elist[],PetscEnum defval)
{
  PetscErrorCode ierr;
  ierr = PetscOptionsGetEnum(prefix,name,elist,&defval,NULL);CHKERRABORT(PETSC_COMM_WORLD,ierr);
  (void)__FUNCT__; return defval;
}

#undef  __FUNCT__
#define __FUNCT__ "IGAGetString"
const char* IGAGetOptString(const char prefix[],const char name[],const char defval[])
{
  PetscErrorCode ierr; static char buffer[1024];
  ierr = PetscStrncpy(buffer,defval,sizeof(buffer));CHKERRABORT(PETSC_COMM_WORLD,ierr);
  ierr = PetscOptionsGetString(prefix,name,buffer,sizeof(buffer),NULL);CHKERRABORT(PETSC_COMM_WORLD,ierr);
  (void)__FUNCT__; return buffer;
}

#undef  __FUNCT__
#define __FUNCT__ "IGAGetOptBool"
PetscBool IGAGetOptBool(const char prefix[],const char name[],PetscBool defval)
{
  PetscErrorCode ierr;
  ierr = PetscOptionsGetBool(prefix,name,&defval,NULL);CHKERRABORT(PETSC_COMM_WORLD,ierr);
  (void)__FUNCT__; return defval;
}

#undef  __FUNCT__
#define __FUNCT__ "IGAGetOptInt"
PetscInt IGAGetOptInt(const char prefix[],const char name[],PetscInt defval)
{
  PetscErrorCode ierr;
  ierr = PetscOptionsGetInt(prefix,name,&defval,NULL);CHKERRABORT(PETSC_COMM_WORLD,ierr);
  (void)__FUNCT__; return defval;
}

#undef  __FUNCT__
#define __FUNCT__ "IGAGetReal"
PetscReal IGAGetOptReal(const char prefix[],const char name[],PetscReal defval)
{
  PetscErrorCode ierr;
  ierr = PetscOptionsGetReal(prefix,name,&defval,NULL);CHKERRABORT(PETSC_COMM_WORLD,ierr);
  (void)__FUNCT__; return defval;
}

#undef  __FUNCT__
#define __FUNCT__ "IGAGetScalar"
PetscScalar IGAGetOptScalar(const char prefix[],const char name[],PetscScalar defval)
{
  PetscErrorCode ierr;
  ierr = PetscOptionsGetScalar(prefix,name,&defval,NULL);CHKERRABORT(PETSC_COMM_WORLD,ierr);
  (void)__FUNCT__; return defval;
}
