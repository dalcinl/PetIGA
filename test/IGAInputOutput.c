#include "petiga.h"

int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  IGA            iga;
  PetscViewer    viewer;
  PetscErrorCode ierr;
  
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);

  ierr = IGAOptionsAlias("-D",  NULL, "-iga_dim");
  ierr = IGAOptionsAlias("-N",  NULL, "-iga_elements");
  ierr = IGAOptionsAlias("-W",  NULL, "-iga_periodic");
  ierr = IGAOptionsAlias("-L",  NULL, "-iga_limits");
  ierr = IGAOptionsAlias("-p",  NULL, "-iga_degree");
  ierr = IGAOptionsAlias("-k",  NULL, "-iga_continuity");

  comm = PETSC_COMM_WORLD;
  ierr = IGACreate(comm,&iga);CHKERRQ(ierr);
  ierr = IGASetFromOptions(iga);CHKERRQ(ierr);
  if (iga->dim < 1) {ierr = IGASetDim(iga,3);CHKERRQ(ierr);}
  ierr = IGASetUp(iga);CHKERRQ(ierr);

  {
    char fieldname[16];PetscInt field,dof;
    ierr = IGAGetDof(iga,&dof);CHKERRQ(ierr);
    for (field=0; field<dof; field++) {
      ierr = PetscSNPrintf(fieldname,sizeof(fieldname),"Field_%D",field);CHKERRQ(ierr);
      ierr = IGASetFieldName(iga,field,fieldname);CHKERRQ(ierr);
    }
  }

  {
    IGA newiga;
    
    ierr = IGAWrite(iga,"iga.dat");CHKERRQ(ierr);
    ierr = IGAWrite(iga,"iga.dat");CHKERRQ(ierr); /* just for testing */
    ierr = IGACreate(comm,&newiga);CHKERRQ(ierr);
    ierr = IGASetOptionsPrefix(newiga,"new_");CHKERRQ(ierr);
    ierr = IGARead(newiga,"iga.dat");CHKERRQ(ierr);
    ierr = IGASetUp(newiga);CHKERRQ(ierr);
    ierr = IGARead(newiga,"iga.dat");CHKERRQ(ierr);  /* just for testing */
    ierr = IGASetUp(newiga);CHKERRQ(ierr);
    ierr = IGADestroy(&newiga);CHKERRQ(ierr);
    
    ierr = PetscViewerBinaryOpen(comm,"iga.dat",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = IGASave(iga,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = IGACreate(comm,&newiga);CHKERRQ(ierr);
    ierr = IGASetOptionsPrefix(newiga,"new_");CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(comm,"iga.dat",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = IGALoad(newiga,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = IGASetUp(newiga);CHKERRQ(ierr);
    ierr = IGADestroy(&newiga);CHKERRQ(ierr);
  }
  
  {
    Vec       vec;
    PetscInt  bs;
    PetscInt  loc;
    PetscReal val;
    
    ierr = IGACreateVec(iga,&vec);CHKERRQ(ierr);
    ierr = VecSet(vec,1.0);CHKERRQ(ierr);
    ierr = IGAWriteVec(iga,vec,"igavec.dat");CHKERRQ(ierr);
    ierr = VecSet(vec,0.0);CHKERRQ(ierr);
    ierr = IGAReadVec (iga,vec,"igavec.dat");CHKERRQ(ierr);
    ierr = VecMin(vec,&loc,&val);CHKERRQ(ierr);
    if ((PetscInt)val != 1) SETERRQ(comm,PETSC_ERR_PLIB,"Loaded Vec does not match");
    ierr = VecMax(vec,&loc,&val);CHKERRQ(ierr);
    if ((PetscInt)val != 1) SETERRQ(comm,PETSC_ERR_PLIB,"Loaded Vec does not match");
    ierr = VecDestroy(&vec);CHKERRQ(ierr);

    ierr = IGAGetDof(iga,&bs);CHKERRQ(ierr);
    ierr = VecCreate(comm,&vec);CHKERRQ(ierr);
    ierr = VecSetBlockSize(vec,bs);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(comm,"igavec.dat",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = VecLoad(vec,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = VecMin(vec,&loc,&val);CHKERRQ(ierr);
    if ((PetscInt)val != 1) SETERRQ(comm,PETSC_ERR_PLIB,"Loaded Vec does not match");
    ierr = VecMax(vec,&loc,&val);CHKERRQ(ierr);
    if ((PetscInt)val != 1) SETERRQ(comm,PETSC_ERR_PLIB,"Loaded Vec does not match");
    ierr = VecDestroy(&vec);CHKERRQ(ierr);

#if !defined(PETSC_USE_COMPLEX)
    ierr = IGACreateVec(iga,&vec);CHKERRQ(ierr);
    ierr = PetscViewerVTKOpen(comm,"igavec.vts",FILE_MODE_WRITE,&viewer);
    ierr = IGADrawVec(iga,vec,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = VecDestroy(&vec);CHKERRQ(ierr);
#endif
  }

  {
    Mat         mat,mat1,mat2;
    Vec         diag,diag1,diag2;
    PetscInt    r,rstart,rend;
    PetscScalar *d;
    PetscBool   match = PETSC_FALSE;

    ierr = IGACreateVec(iga,&diag);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(diag,&rstart,&rend);CHKERRQ(ierr);
    ierr = VecGetArray(diag,&d);CHKERRQ(ierr);
    for (r=rstart; r<rend; r++) d[r-rstart] = (PetscReal)r;
    ierr = VecRestoreArray(diag,&d);CHKERRQ(ierr);

    ierr = VecDuplicate(diag,&diag1);CHKERRQ(ierr);
    ierr = VecDuplicate(diag,&diag2);CHKERRQ(ierr);

    ierr = IGACreateMat(iga,&mat);CHKERRQ(ierr);
    ierr = MatDiagonalSet(mat,diag,INSERT_VALUES);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(comm,"igamat.dat",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    /*ierr = PetscViewerSetFormat(viewer,PETSC_VIEWER_NATIVE);CHKERRQ(ierr);*/
    ierr = MatView(mat,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = MatDestroy(&mat);CHKERRQ(ierr);

    ierr = IGACreateMat(iga,&mat1);CHKERRQ(ierr);
    ierr = MatDuplicate(mat1,MAT_DO_NOT_COPY_VALUES,&mat2);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(comm,"igamat.dat",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = MatLoad(mat1,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = MatGetDiagonal(mat1,diag1);CHKERRQ(ierr);
    ierr = MatDestroy(&mat1);CHKERRQ(ierr);
    ierr = VecEqual(diag,diag1,&match);;CHKERRQ(ierr);
    if (!match) SETERRQ(comm,PETSC_ERR_PLIB,"Loaded Mat does not match");
    ierr = VecDestroy(&diag1);CHKERRQ(ierr);

    ierr = PetscViewerBinaryOpen(comm,"igamat.dat",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = MatLoad(mat2,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = MatGetDiagonal(mat2,diag2);CHKERRQ(ierr);
    ierr = MatDestroy(&mat2);CHKERRQ(ierr);
    ierr = VecEqual(diag,diag2,&match);;CHKERRQ(ierr);
    if (!match) SETERRQ(comm,PETSC_ERR_PLIB,"Loaded Mat does not match");
    ierr = VecDestroy(&diag2);CHKERRQ(ierr);

    ierr = VecDestroy(&diag);CHKERRQ(ierr);
  }

  ierr = IGADestroy(&iga);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
 }
