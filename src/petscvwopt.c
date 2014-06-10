#include <petsc.h>
#if PETSC_VERSION_LE(3,3,0)

PETSC_EXTERN PetscErrorCode PetscOptionsGetViewer(MPI_Comm,const char[],const char[],PetscViewer*,PetscViewerFormat*,PetscBool*);

#undef __FUNCT__
#define __FUNCT__ "PetscEListFind"
static PetscErrorCode PetscEListFind(PetscInt n,const char *const *list,const char *str,PetscInt *value,PetscBool *found)
{
  PetscInt  i;
  PetscBool matched;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (found) *found = PETSC_FALSE;
  for (i=0; i<n; i++) {
    ierr = PetscStrcasecmp(str,list[i],&matched);CHKERRQ(ierr);
    if (matched || !str[0]) {
      if (found) *found = PETSC_TRUE;
      *value = i;
      break;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscEnumFind"
static PetscErrorCode PetscEnumFind(const char *const *enumlist,const char *str,PetscEnum *value,PetscBool *found)
{
  PetscInt       n,evalue;
  PetscBool      efound;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  for (n = 0; enumlist[n]; n++) {
    if (n > 50) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"List argument appears to be wrong or have more than 50 entries");
  }
  if (n < 3) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"List argument must have at least two entries: typename and type prefix");
  n -= 3;                       /* drop enum name, prefix, and null termination */
  ierr = PetscEListFind(n,enumlist,str,&evalue,&efound);CHKERRQ(ierr);
  if (efound) *value = (PetscEnum)evalue;
  if (found ) *found = efound;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscOptionsGetViewer"
PetscErrorCode PetscOptionsGetViewer(MPI_Comm comm,const char pre[],const char name[],PetscViewer *viewer,PetscViewerFormat *format,PetscBool *set)
{
  char           value[2*PETSC_MAX_PATH_LEN];
  PetscBool      flag;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidCharPointer(name,3);
  if (format) *format = PETSC_VIEWER_DEFAULT;
  if (set) *set = PETSC_FALSE;
  ierr = PetscOptionsGetString(pre,name,value,sizeof(value),&flag);CHKERRQ(ierr);
  if (flag) {
    if (set) *set = PETSC_TRUE;
    if (!value[0]) {
      ierr = PetscViewerASCIIGetStdout(comm,viewer);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)*viewer);CHKERRQ(ierr);
    } else {
      char       *loc0_vtype,*loc1_fname,*loc2_fmt = NULL,*loc3_fmode = NULL;
      PetscInt   cnt;
      const char *const PetscFileModes[] = {"READ","WRITE","APPEND","UPDATE","APPEND_UPDATE","PetscFileMode","PETSC_FILE_",0};
      const char *const viewers[] = {PETSCVIEWERASCII,PETSCVIEWERBINARY,PETSCVIEWERDRAW,PETSCVIEWERSOCKET,PETSCVIEWERMATLAB,PETSCVIEWERAMS,PETSCVIEWERVTK,0};
      ierr = PetscStrallocpy(value,&loc0_vtype);CHKERRQ(ierr);
      ierr = PetscStrchr(loc0_vtype,':',&loc1_fname);CHKERRQ(ierr);
      if (loc1_fname) {
        *loc1_fname++ = 0;
        ierr = PetscStrchr(loc1_fname,':',&loc2_fmt);CHKERRQ(ierr);
      }
      if (loc2_fmt) {
        *loc2_fmt++ = 0;
        ierr = PetscStrchr(loc2_fmt,':',&loc3_fmode);CHKERRQ(ierr);
      }
      if (loc3_fmode) *loc3_fmode++ = 0;
      ierr = PetscStrendswithwhich(*loc0_vtype ? loc0_vtype : "ascii",viewers,&cnt);CHKERRQ(ierr);
      if (cnt > (PetscInt) sizeof(viewers)-1) SETERRQ1(comm,PETSC_ERR_ARG_OUTOFRANGE,"Unknown viewer type: %s",loc0_vtype);
      if (!loc1_fname) {
        switch (cnt) {
        case 0:
          ierr = PetscViewerASCIIGetStdout(comm,viewer);CHKERRQ(ierr);
          break;
        case 1:
          if (!(*viewer = PETSC_VIEWER_BINARY_(comm))) CHKERRQ(PETSC_ERR_PLIB);
          break;
        case 2:
          if (!(*viewer = PETSC_VIEWER_DRAW_(comm))) CHKERRQ(PETSC_ERR_PLIB);
          break;
#if defined(PETSC_USE_SOCKET_VIEWER)
        case 3:
          if (!(*viewer = PETSC_VIEWER_SOCKET_(comm))) CHKERRQ(PETSC_ERR_PLIB);
          break;
#endif
#if defined(PETSC_HAVE_MATLAB_ENGINE)
        case 4:
          if (!(*viewer = PETSC_VIEWER_MATLAB_(comm))) CHKERRQ(PETSC_ERR_PLIB);
          break;
#endif
#if defined(PETSC_HAVE_AMS)
        case 5:
          if (!(*viewer = PETSC_VIEWER_AMS_(comm))) CHKERRQ(PETSC_ERR_PLIB);
          break;
#endif
        default: SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unsupported viewer %s",loc0_vtype);
        }
        ierr = PetscObjectReference((PetscObject)*viewer);CHKERRQ(ierr);
      } else {
        if (loc2_fmt && !*loc1_fname && (cnt == 0)) { /* ASCII format without file name */
          ierr = PetscViewerASCIIGetStdout(comm,viewer);CHKERRQ(ierr);
          ierr = PetscObjectReference((PetscObject)*viewer);CHKERRQ(ierr);
        } else {
          PetscFileMode fmode;
          ierr = PetscViewerCreate(comm,viewer);CHKERRQ(ierr);
          ierr = PetscViewerSetType(*viewer,*loc0_vtype ? loc0_vtype : "ascii");CHKERRQ(ierr);
#if defined(PETSC_HAVE_AMS)
          ierr = PetscViewerAMSSetCommName(*viewer,loc1_fname);CHKERRQ(ierr);
#endif
          fmode = FILE_MODE_WRITE;
          if (loc3_fmode && *loc3_fmode) { /* Has non-empty file mode ("write" or "append") */
            ierr = PetscEnumFind(PetscFileModes,loc3_fmode,(PetscEnum*)&fmode,&flag);CHKERRQ(ierr);
            if (!flag) SETERRQ1(comm,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown file mode: %s",loc3_fmode);
          }
          ierr = PetscViewerFileSetMode(*viewer,flag?fmode:FILE_MODE_WRITE);CHKERRQ(ierr);
          ierr = PetscViewerFileSetName(*viewer,loc1_fname);CHKERRQ(ierr);
        }
      }
      if (loc2_fmt && *loc2_fmt) {
        ierr = PetscEnumFind(PetscViewerFormats,loc2_fmt,(PetscEnum*)format,&flag);CHKERRQ(ierr);
        if (!flag) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unknown viewer format %s",loc2_fmt);CHKERRQ(ierr);
        if (((int)*format) >= 5) *format = (PetscViewerFormat)(((int)*format)-2); /**/
      }
      ierr = PetscViewerSetUp(*viewer);CHKERRQ(ierr);
      ierr = PetscFree(loc0_vtype);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#endif
