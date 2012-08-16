! -*- f90 -*-

#include "petscconf.h"

#define C_PETSC_ERRORCODE C_INT

#if defined(PETSC_USE_64BIT_INDICES)
#define C_PETSC_INT C_LONG_LONG
#else
#define C_PETSC_INT C_INT
#endif

#if defined(PETSC_USE_REAL_SINGLE)
#define C_PETSC_REAL    C_FLOAT
#define C_PETSC_COMPLEX C_FLOAT_COMPLEX
#elif defined(PETSC_USE_REAL_DOUBLE)
#define C_PETSC_REAL    C_DOUBLE
#define C_PETSC_COMPLEX C_DOUBLE_COMPLEX
#elif defined(PETSC_USE_REAL_LONG_DOUBLE)
#define C_PETSC_REAL    C_LONG_DOUBLE
#define C_PETSC_COMPLEX C_LONG_DOUBLE_COMPLEX
#elif defined(PETSC_USE_REAL___FLOAT128)
#define C_PETSC_REAL    C_FLOAT128
#define C_PETSC_COMPLEX C_FLOAT128_COMPLEX
#endif

#if defined(PETSC_USE_COMPLEX)
#define C_PETSC_SCALAR  C_PETSC_COMPLEX
#else
#define C_PETSC_SCALAR  C_PETSC_REAL
#endif

#if defined(PETSC_USE_COMPLEX)
#define scalar complex
#else
#define scalar real
#endif

module PetIGA

  use ISO_C_BINDING, only: IGA_ERRCODE => C_PETSC_ERRORCODE
  use ISO_C_BINDING, only: C_PTR

  use ISO_C_BINDING, only: IGA_INT     => C_PETSC_INT
  use ISO_C_BINDING, only: IGA_REAL    => C_PETSC_REAL
  use ISO_C_BINDING, only: IGA_SCALAR  => C_PETSC_SCALAR
  use ISO_C_BINDING, only: IGA_COMPLEX => C_PETSC_COMPLEX
  implicit none

  !type, bind(C) :: IGA
  !   private
  !end type IGAElement

  !type, bind(C) :: IGAElement
  !   integer(kind=IGA_INT) :: private_refct
  !   integer(kind=IGA_INT) :: private_start(3);
  !   integer(kind=IGA_INT) :: private_width(3);
  !   integer(kind=IGA_INT) :: ID(3);
  !   integer(kind=IGA_INT) :: count
  !   integer(kind=IGA_INT) :: index
  !   integer(kind=IGA_INT) :: qnp
  !   integer(kind=IGA_INT) :: nen
  !   integer(kind=IGA_INT) :: dof
  !   integer(kind=IGA_INT) :: dim
  !   integer(kind=IGA_INT) :: nsd
  !end type IGAElement

  type, bind(C) :: IGAPoint
     integer(kind=IGA_INT) :: private_refct
     integer(kind=IGA_INT) :: count
     integer(kind=IGA_INT) :: index
     integer(kind=IGA_INT) :: nen
     integer(kind=IGA_INT) :: dof
     integer(kind=IGA_INT) :: dim
     integer(kind=IGA_INT) :: nsd
  end type IGAPoint

  interface

     integer(kind=IGA_ERRCODE) &
     function IGAPointGetIndex(p,index) bind(C)
       import; implicit none
       type(IGAPoint), intent(in) :: p
       integer(kind=IGA_INT), intent(out) :: index
     end function IGAPointGetIndex

     integer(kind=IGA_ERRCODE) &
     function IGAPointGetCount(p,count) bind(C)
       import; implicit none
       type(IGAPoint), intent(in) :: p
       integer(kind=IGA_INT), intent(out) :: count
     end function IGAPointGetCount

     integer(kind=IGA_ERRCODE) &
     function IGAPointFormPoint(p,x) bind(C)
       import; implicit none
       type(IGAPoint), intent(in) :: p
       real(kind=IGA_REAL), intent(out) :: x(p%dim)
     end function IGAPointFormPoint

     integer(kind=IGA_ERRCODE) &
     function IGAPointFormGradMap(p,map,inv) bind(C)
       import; implicit none
       type(IGAPoint), intent(in) :: p
       real(kind=IGA_REAL), intent(out) :: map(p%dim,p%dim)
       real(kind=IGA_REAL), intent(out) :: inv(p%dim,p%dim)
     end function IGAPointFormGradMap

     integer(kind=IGA_ERRCODE) &
     function IGAPointFormShapeFuns(p,der,N) bind(C)
       import; implicit none
       type(IGAPoint), intent(in) :: p
       integer(kind=IGA_INT), intent(in),value :: der
       real   (kind=IGA_REAL), intent(out) :: N(p%dim**der,p%nen)
     end function IGAPointFormShapeFuns

  end interface

  interface IGAPointFormValue
     integer(kind=IGA_ERRCODE) &
     function IGAPointFormValue(p,U,v) bind(C)
       import; implicit none
       type(IGAPoint), intent(in) :: p
       scalar(kind=IGA_SCALAR), intent(in)  :: U(p%dof,p%nen)
       scalar(kind=IGA_SCALAR), intent(out) :: v(p%dof)
     end function IGAPointFormValue
     module procedure IGAPointFormValue_S
  end interface IGAPointFormValue

  interface IGAPointFormGrad
     integer(kind=IGA_ERRCODE) &
     function IGAPointFormGrad(p,U,v) bind(C)
       import; implicit none
       type(IGAPoint), intent(in) :: p
       scalar(kind=IGA_SCALAR), intent(in)  :: U(p%dof,p%nen)
       scalar(kind=IGA_SCALAR), intent(out) :: v(p%dim,p%dof)
     end function IGAPointFormGrad
     module procedure IGAPointFormGrad_S
  end interface IGAPointFormGrad

  interface IGAPointFormHess
     integer(kind=IGA_ERRCODE) &
     function IGAPointFormHess(p,U,v) bind(C)
       import; implicit none
       integer(kind=IGA_ERRCODE)     :: ierr
       type(IGAPoint), intent(in) :: p
       scalar(kind=IGA_SCALAR), intent(in)  :: U(p%dof,p%nen)
       scalar(kind=IGA_SCALAR), intent(out) :: v(p%dim,p%dim,p%dof)
     end function IGAPointFormHess
     module procedure IGAPointFormHess_S
  end interface IGAPointFormHess

  interface IGAPointFormDer0
     module procedure IGAPointFormValue_V
     module procedure IGAPointFormValue_S
  end interface IGAPointFormDer0

  interface IGAPointFormDer1
     module procedure IGAPointFormGrad_V
     module procedure IGAPointFormGrad_S
  end interface IGAPointFormDer1

  interface IGAPointFormDer2
     module procedure IGAPointFormHess_V
     module procedure IGAPointFormHess_S
  end interface IGAPointFormDer2

  interface IGAPointFormDer3
     integer(kind=IGA_ERRCODE) &
     function IGAPointFormDer3(p,U,v) bind(C)
       import; implicit none
       type(IGAPoint), intent(in) :: p
       scalar(kind=IGA_SCALAR), intent(in)  :: U(p%dof,p%nen)
       scalar(kind=IGA_SCALAR), intent(out) :: v(p%dim,p%dim,p%dim,p%dof)
     end function IGAPointFormDer3
     module procedure IGAPointFormDer3_S
  end interface IGAPointFormDer3

  contains

    integer(kind=IGA_ERRCODE) &
    function IGAPointFormValue_S(p,U,v) bind(C) result (ierr) 
      implicit none
      type(IGAPoint), intent(in) :: p
      scalar(kind=IGA_SCALAR), intent(in)  :: U(p%nen)
      scalar(kind=IGA_SCALAR), intent(out) :: v
      scalar(kind=IGA_SCALAR)  :: tmp(1)
      ierr = IGAPointFormValue(p, reshape(U,(/1,p%nen/)), tmp)
      v = tmp(1)
    end function IGAPointFormValue_S

    integer(kind=IGA_ERRCODE) &
    function IGAPointFormValue_V(p,U,v) bind(C) result (ierr)
      implicit none
      type(IGAPoint), intent(in) :: p
      scalar(kind=IGA_SCALAR), intent(in)  :: U(p%dof,p%nen)
      scalar(kind=IGA_SCALAR), intent(out) :: v(p%dof)
      ierr = IGAPointFormValue(p,U,v)
    end function IGAPointFormValue_V

    integer(kind=IGA_ERRCODE) &
    function IGAPointFormGrad_S(p,U,v) bind(C) result (ierr)
      implicit none
      type(IGAPoint), intent(in) :: p
      scalar(kind=IGA_SCALAR), intent(in)  :: U(p%nen)
      scalar(kind=IGA_SCALAR), intent(out) :: v(p%dim)
      scalar(kind=IGA_SCALAR)  :: tmp(p%dim,1)
      ierr = IGAPointFormGrad_V(p, reshape(U,(/1,p%nen/)), tmp)
      v = tmp(:,1)
    end function IGAPointFormGrad_S

    integer(kind=IGA_ERRCODE) &
    function IGAPointFormGrad_V(p,U,v) bind(C) result (ierr)
      implicit none
      type(IGAPoint), intent(in) :: p
      scalar(kind=IGA_SCALAR), intent(in)  :: U(p%dof,p%nen)
      scalar(kind=IGA_SCALAR), intent(out) :: v(p%dim,p%dof)
      ierr = IGAPointFormGrad(p,U,v)
    end function IGAPointFormGrad_V

    integer(kind=IGA_ERRCODE) &
    function IGAPointFormHess_S(p,U,v) bind(C) result (ierr)
      implicit none
      type(IGAPoint), intent(in) :: p
      scalar(kind=IGA_SCALAR), intent(in)  :: U(p%nen)
      scalar(kind=IGA_SCALAR), intent(out) :: v(p%dim,p%dim)
      scalar(kind=IGA_SCALAR)  :: tmp(p%dim,p%dim,1)
      ierr = IGAPointFormHess(p, reshape(U,(/1,p%nen/)), tmp)
      v = tmp(:,:,1)
    end function IGAPointFormHess_S

    integer(kind=IGA_ERRCODE) &
    function IGAPointFormHess_V(p,U,v) bind(C) result (ierr)
      implicit none
      type(IGAPoint), intent(in) :: p
      scalar(kind=IGA_SCALAR), intent(in)  :: U(p%dof,p%nen)
      scalar(kind=IGA_SCALAR), intent(out) :: v(p%dim,p%dim,p%dof)
      scalar(kind=IGA_SCALAR)  :: tmp(p%dim,p%dim,1)
      ierr = IGAPointFormHess(p,U,v)
    end function IGAPointFormHess_V

    integer(kind=IGA_ERRCODE) &
    function IGAPointFormDer3_S(p,U,v) bind(C) result (ierr)
      implicit none
      type(IGAPoint), intent(in) :: p
      scalar(kind=IGA_SCALAR), intent(in)  :: U(p%nen)
      scalar(kind=IGA_SCALAR), intent(out) :: v(p%dim,p%dim,p%dim)
      scalar(kind=IGA_SCALAR)  :: tmp(p%dim,p%dim,p%dim,1)
      ierr = IGAPointFormDer3(p, reshape(U,(/1,p%nen/)), tmp)
      v = tmp(:,:,:,1)
    end function IGAPointFormDer3_S

    integer(kind=IGA_ERRCODE) &
    function IGAPointFormDer3_V(p,U,v) bind(C) result (ierr)
      implicit none
      type(IGAPoint), intent(in) :: p
      scalar(kind=IGA_SCALAR), intent(in)  :: U(p%dof,p%nen)
      scalar(kind=IGA_SCALAR), intent(out) :: v(p%dim,p%dim,p%dim,p%dof)
      ierr = IGAPointFormDer3(p,U,v)
    end function IGAPointFormDer3_V

end module PetIGA
