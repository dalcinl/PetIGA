#include "petiga.h"

#if PETSC_VERSION_LT(3,7,0)
#define PetscOptionsHasName(op,pr,nm,set)       PetscOptionsHasName(pr,nm,set)
#define PetscOptionsSetValue(op,nm,vl)          PetscOptionsSetValue(nm,vl)
#define PetscOptionsPrefixPush(op,pr)           PetscOptionsPrefixPush(pr)
#define PetscOptionsPrefixPop(op)               PetscOptionsPrefixPop()
#define PetscOptionsClearValue(op,nm)           PetscOptionsClearValue(nm)
#define PetscOptionsGetBool(op,pr,nm,vl,set)    PetscOptionsGetBool(pr,nm,vl,set)
#define PetscOptionsGetEnum(op,pr,nm,el,dv,set) PetscOptionsGetEnum(pr,nm,el,dv,set)
#define PetscOptionsGetInt(op,pr,nm,vl,set)     PetscOptionsGetInt(pr,nm,vl,set)
#define PetscOptionsGetReal(op,pr,nm,vl,set)    PetscOptionsGetReal(pr,nm,vl,set)
#define PetscOptionsGetScalar(op,pr,nm,vl,set)  PetscOptionsGetScalar(pr,nm,vl,set)
#define PetscOptionsGetString(op,pr,nm,s,n,set) PetscOptionsGetString(pr,nm,s,n,set)
#endif

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

#undef  __FUNCT__
#define __FUNCT__ "IGAOptionsDefault"
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

#undef  __FUNCT__
#define __FUNCT__ "IGAOptionsReject"
PetscErrorCode IGAOptionsReject(const char prefix[],const char name[])
{
  PetscBool      flag = PETSC_FALSE;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscOptionsHasName(NULL,prefix,name,&flag);CHKERRQ(ierr);
  if (flag) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Disabled option: %s",name);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAGetOptEnum"
PetscEnum IGAGetOptEnum(const char prefix[],const char name[],const char *const elist[],PetscEnum defval)
{
  PetscErrorCode ierr;
  ierr = PetscOptionsGetEnum(NULL,prefix,name,elist,&defval,NULL);CHKERRABORT(PETSC_COMM_WORLD,ierr);
  (void)__FUNCT__; return defval;
}

#undef  __FUNCT__
#define __FUNCT__ "IGAGetString"
const char* IGAGetOptString(const char prefix[],const char name[],const char defval[])
{
  PetscErrorCode ierr; static char buffer[1024];
  ierr = PetscStrncpy(buffer,defval,sizeof(buffer));CHKERRABORT(PETSC_COMM_WORLD,ierr);
  ierr = PetscOptionsGetString(NULL,prefix,name,buffer,sizeof(buffer),NULL);CHKERRABORT(PETSC_COMM_WORLD,ierr);
  (void)__FUNCT__; return buffer;
}

#undef  __FUNCT__
#define __FUNCT__ "IGAGetOptBool"
PetscBool IGAGetOptBool(const char prefix[],const char name[],PetscBool defval)
{
  PetscErrorCode ierr;
  ierr = PetscOptionsGetBool(NULL,prefix,name,&defval,NULL);CHKERRABORT(PETSC_COMM_WORLD,ierr);
  (void)__FUNCT__; return defval;
}

#undef  __FUNCT__
#define __FUNCT__ "IGAGetOptInt"
PetscInt IGAGetOptInt(const char prefix[],const char name[],PetscInt defval)
{
  PetscErrorCode ierr;
  ierr = PetscOptionsGetInt(NULL,prefix,name,&defval,NULL);CHKERRABORT(PETSC_COMM_WORLD,ierr);
  (void)__FUNCT__; return defval;
}

#undef  __FUNCT__
#define __FUNCT__ "IGAGetReal"
PetscReal IGAGetOptReal(const char prefix[],const char name[],PetscReal defval)
{
  PetscErrorCode ierr;
  ierr = PetscOptionsGetReal(NULL,prefix,name,&defval,NULL);CHKERRABORT(PETSC_COMM_WORLD,ierr);
  (void)__FUNCT__; return defval;
}

#undef  __FUNCT__
#define __FUNCT__ "IGAGetScalar"
PetscScalar IGAGetOptScalar(const char prefix[],const char name[],PetscScalar defval)
{
  PetscErrorCode ierr;
  ierr = PetscOptionsGetScalar(NULL,prefix,name,&defval,NULL);CHKERRABORT(PETSC_COMM_WORLD,ierr);
  (void)__FUNCT__; return defval;
}


#if PETSC_VERSION_LT(3,6,0)

PETSC_EXTERN PetscErrorCode PetscOptionsGetEnumArray(const char[],const char[],const char * const *,PetscEnum[],PetscInt*,PetscBool*);
PETSC_EXTERN PetscErrorCode PetscOptionsEnumArray(const char[],const char[],const char[],const char *const *list,PetscEnum[],PetscInt*,PetscBool*);
extern PetscOptionsObjectType PetscOptionsObject;
#define ManSection(str) ((str) ? (str) : "None")

#undef  __FUNCT__
#define __FUNCT__ "PetscOptionsGetEnumArray"
PetscErrorCode PetscOptionsGetEnumArray(const char pre[],const char name[],const char *const *list,PetscEnum dvalue[],PetscInt *nmax,PetscBool *set)
{
  char           *svalue;
  PetscInt       nlist = 0,n = 0;
  PetscInt       ivalue;
  PetscBool      flag;
  PetscToken     token;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(name,2);
  PetscValidPointer(list,3);
  PetscValidIntPointer(dvalue,4);
  PetscValidIntPointer(nmax,5);

  while(list[nlist++]) if (nlist > 50) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"List argument appears to be wrong or have more than 50 entries");
  if (nlist < 3) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"List argument must have at least two entries: typename and type prefix");
  nlist -= 3; /* drop enum name, prefix, and null termination */

  ierr = PetscOptionsFindPair_Private(pre,name,&svalue,&flag);CHKERRQ(ierr);
  if (!flag) {
    if (set) *set = PETSC_FALSE;
    *nmax = 0;
    PetscFunctionReturn(0);
  }
  if (!svalue) {
    if (set) *set = PETSC_TRUE;
    *nmax = 0;
    PetscFunctionReturn(0);
  }
  if (set) *set = PETSC_TRUE;

  ierr = PetscTokenCreate(svalue,',',&token);CHKERRQ(ierr);
  ierr = PetscTokenFind(token,&svalue);CHKERRQ(ierr);
  while (svalue && n < *nmax) {
    ierr = PetscEListFind(nlist,list,svalue,&ivalue,&flag);CHKERRQ(ierr);
    if (!flag) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_USER,"Unknown enum value '%s' for -%s%s",svalue,pre?pre:"",name+1);
    dvalue[n++] = (PetscEnum)ivalue;
    ierr = PetscTokenFind(token,&svalue);CHKERRQ(ierr);
  }
  ierr  = PetscTokenDestroy(&token);CHKERRQ(ierr);
  *nmax = n;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "PetscOptionsEnumArray"
PetscErrorCode PetscOptionsEnumArray(const char opt[],const char text[],const char man[],const char *const *list,PetscEnum value[],PetscInt *n,PetscBool *set)
{
  PetscInt       i,nlist = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  while (list[nlist++]) if (nlist > 50) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"List argument appears to be wrong or have more than 50 entries");
  if (nlist < 3) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"List argument must have at least two entries: typename and type prefix");
  nlist -= 3; /* drop enum name, prefix, and null termination */
  ierr = PetscOptionsGetEnumArray(PetscOptionsObject.prefix,opt,list,value,n,set);CHKERRQ(ierr);
  if (PetscOptionsObject.printhelp && PetscOptionsPublishCount == 1 && !PetscOptionsObject.alreadyprinted) {
    ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,"  -%s%s <%s",PetscOptionsObject.prefix?PetscOptionsObject.prefix :"",opt+1,list[value[0]]);CHKERRQ(ierr);
    for (i=1; i<*n; i++) {ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,",%s",list[value[i]]);CHKERRQ(ierr);}
    ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm,">: %s (choose one of)",text);CHKERRQ(ierr);
    for (i=0; i<nlist; i++) {ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm," %s",list[i]);CHKERRQ(ierr);}
    ierr = (*PetscHelpPrintf)(PetscOptionsObject.comm," (%s)\n",ManSection(man));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#endif
