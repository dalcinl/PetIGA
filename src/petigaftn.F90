! -*- f90 -*-

#include "petscconf.h"

#define C_PETSC_ERRORCODE C_INT
#define C_PETSC_BOOL C_INT

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
#define C_PETSC_SCALAR C_PETSC_COMPLEX
#else
#define C_PETSC_SCALAR C_PETSC_REAL
#endif

#if defined(PETSC_USE_COMPLEX)
#define scalar complex
#else
#define scalar real
#endif

module PetIGA
  use ISO_C_BINDING, only: C_PTR
  use ISO_C_BINDING, only: IGA_ERRCODE_KIND => C_PETSC_ERRORCODE
  use ISO_C_BINDING, only: IGA_LOGICAL_KIND => C_PETSC_BOOL
  use ISO_C_BINDING, only: IGA_INTEGER_KIND => C_PETSC_INT
  use ISO_C_BINDING, only: IGA_REAL_KIND    => C_PETSC_REAL
  use ISO_C_BINDING, only: IGA_COMPLEX_KIND => C_PETSC_COMPLEX
  use ISO_C_BINDING, only: IGA_SCALAR_KIND  => C_PETSC_SCALAR
  implicit none

  !type, bind(C) :: IGA
  !   private
  !end type IGAElement

  !type, bind(C) :: IGAElement
  !   integer(kind=IGA_INTEGER_KIND) :: private_refct
  !   integer(kind=IGA_INTEGER_KIND) :: private_start(3);
  !   integer(kind=IGA_INTEGER_KIND) :: private_width(3);
  !   integer(kind=IGA_INTEGER_KIND) :: private_ID(3);
  !   integer(kind=IGA_INTEGER_KIND) :: count
  !   integer(kind=IGA_INTEGER_KIND) :: index
  !   integer(kind=IGA_INTEGER_KIND) :: qnp
  !   integer(kind=IGA_INTEGER_KIND) :: nen
  !   integer(kind=IGA_INTEGER_KIND) :: dof
  !   integer(kind=IGA_INTEGER_KIND) :: dim
  !   integer(kind=IGA_INTEGER_KIND) :: nsd
  !end type IGAElement

  type, bind(C) :: IGAPoint
     integer(kind=IGA_INTEGER_KIND) :: private_refct

     integer(kind=IGA_LOGICAL_KIND) :: atboundary
     integer(kind=IGA_INTEGER_KIND) :: boundary_id

     integer(kind=IGA_INTEGER_KIND) :: count
     integer(kind=IGA_INTEGER_KIND) :: index

     integer(kind=IGA_INTEGER_KIND) :: neq
     integer(kind=IGA_INTEGER_KIND) :: nen
     integer(kind=IGA_INTEGER_KIND) :: dof
     integer(kind=IGA_INTEGER_KIND) :: dim
     integer(kind=IGA_INTEGER_KIND) :: nsd
     integer(kind=IGA_INTEGER_KIND) :: npd

     type(C_PTR) :: rational
     type(C_PTR) :: geometry
     type(C_PTR) :: property

     type(C_PTR) :: weight
     type(C_PTR) :: detJac

     type(C_PTR) :: point
     type(C_PTR) :: normal
     
     type(C_PTR) :: basis(0:4)
     type(C_PTR) :: shape(0:4)
     
     type(C_PTR) :: mapU(0:4)
     type(C_PTR) :: mapX(0:4)
     type(C_PTR) :: detX
     type(C_PTR) :: detS

     type(C_PTR)                    :: private_parent
     integer(kind=IGA_INTEGER_KIND) :: private_nvec
     type(C_PTR)                    :: private_wvec(8)
     integer(kind=IGA_INTEGER_KIND) :: private_nmat
     type(C_PTR)                    :: private_wmat(4)
  end type IGAPoint


  interface IGA_GeomMap
     module procedure IGA_GeomMap
  end interface IGA_GeomMap

  interface IGA_GradGeomMap
     module procedure IGA_GradGeomMap
  end interface IGA_GradGeomMap

  !interface IGA_HessGeomMap
  !   module procedure IGA_HessGeomMap
  !end interface IGA_HessGeomMap

  interface IGA_InvGradGeomMap
     module procedure IGA_InvGradGeomMap
  end interface IGA_InvGradGeomMap


  interface IGA_Basis0
     module procedure IGA_Basis0
  end interface IGA_Basis0

  interface IGA_Basis1
     module procedure IGA_Basis1
  end interface IGA_Basis1

  interface IGA_Basis2
     module procedure IGA_Basis2
  end interface IGA_Basis2

  interface IGA_Basis3
     module procedure IGA_Basis3
  end interface IGA_Basis3

  interface IGA_Basis4
     module procedure IGA_Basis4
  end interface IGA_Basis4


  interface IGA_Shape0
     module procedure IGA_Shape0
  end interface IGA_Shape0

  interface IGA_Shape1
     module procedure IGA_Shape1
  end interface IGA_Shape1

  interface IGA_Shape2
     module procedure IGA_Shape2
  end interface IGA_Shape2

  interface IGA_Shape3
     module procedure IGA_Shape3
  end interface IGA_Shape3

  interface IGA_Shape4
     module procedure IGA_Shape4
  end interface IGA_Shape4


  interface IGA_Eval
     module procedure IGA_Shape_Der0_S
     module procedure IGA_Shape_Der0_V
     module procedure IGA_Shape_Der1_S
     module procedure IGA_Shape_Der1_V
     module procedure IGA_Shape_Der2_S
     module procedure IGA_Shape_Der2_V
     module procedure IGA_Shape_Der3_S
     module procedure IGA_Shape_Der3_V
     module procedure IGA_Shape_Der4_S
     module procedure IGA_Shape_Der4_V
  end interface IGA_Eval

  interface IGA_Value
     module procedure IGA_Shape_Der0_S
     module procedure IGA_Shape_Der0_V
     module procedure IGA_Point_Der0_S
     module procedure IGA_Point_Der0_V
  end interface IGA_Value

  interface IGA_Grad
     module procedure IGA_Shape_Der1_S
     module procedure IGA_Shape_Der1_V
     module procedure IGA_Point_Der1_S
     module procedure IGA_Point_Der1_V
  end interface IGA_Grad

  interface IGA_Hess
     module procedure IGA_Shape_Der2_S
     module procedure IGA_Shape_Der2_V
     module procedure IGA_Point_Der2_S
     module procedure IGA_Point_Der2_V
  end interface IGA_Hess

  interface IGA_Der3
     module procedure IGA_Shape_Der3_S
     module procedure IGA_Shape_Der3_V
     module procedure IGA_Point_Der3_S
     module procedure IGA_Point_Der3_V
  end interface IGA_Der3

  interface IGA_Der4
     module procedure IGA_Shape_Der4_S
     module procedure IGA_Shape_Der4_V
     module procedure IGA_Point_Der4_S
     module procedure IGA_Point_Der4_V
  end interface IGA_Der4

  interface IGA_Div
     module procedure IGA_Shape_Div
     module procedure IGA_Point_Div
  end interface IGA_Div

  interface IGA_Del2
     module procedure IGA_Shape_Del2_S
     module procedure IGA_Shape_Del2_V
     module procedure IGA_Point_Del2_S
     module procedure IGA_Point_Del2_V
  end interface IGA_Del2

  contains

    function IGA_Rational(p) result(W)
      use ISO_C_BINDING, only: c2f => C_F_POINTER
      use ISO_C_BINDING, only: nonnull => C_ASSOCIATED
      implicit none
      type(IGAPoint), intent(in) :: p
      real(kind=IGA_REAL_KIND), pointer :: W(:)
      nullify(W)
      if (nonnull(p%rational)) &
      call c2f(p%rational,W,(/p%nen/))
    end function IGA_Rational

    function IGA_Geometry(p) result(X)
      use ISO_C_BINDING, only: c2f => C_F_POINTER
      use ISO_C_BINDING, only: nonnull => C_ASSOCIATED
      implicit none
      type(IGAPoint), intent(in) :: p
      real(kind=IGA_REAL_KIND), pointer :: X(:,:)
      nullify(X)
      if (nonnull(p%geometry)) &
      call c2f(p%geometry,X,(/p%nsd,p%nen/))
    end function IGA_Geometry

    function IGA_Property(p) result(A)
      use ISO_C_BINDING, only: c2f => C_F_POINTER
      use ISO_C_BINDING, only: nonnull => C_ASSOCIATED
      implicit none
      type(IGAPoint), intent(in) :: p
      real(kind=IGA_REAL_KIND), pointer :: A(:,:)
      nullify(A)
      if (nonnull(p%property)) &
      call c2f(p%property,A,(/p%npd,p%nen/))
    end function IGA_Property

    function IGA_GeomMap(p) result (X)
      implicit none
      type(IGAPoint), intent(in) :: p
      real(kind=IGA_REAL_KIND)   :: X(p%nsd)
      interface
         subroutine IGAPoint_GeomMap(p,X) bind(C)
           import; implicit none;
           type(IGAPoint), intent(in) :: p
           real(kind=IGA_REAL_KIND), intent(out) :: X(p%nsd)
         end subroutine IGAPoint_GeomMap
      end interface
      call IGAPoint_GeomMap(p,X)
    end function IGA_GeomMap

    function IGA_GradGeomMap(p) result (F)
      implicit none
      type(IGAPoint), intent(in) :: p
      real(kind=IGA_REAL_KIND)   :: F(p%dim,p%nsd)
      interface
         subroutine IGAPoint_GradGeomMap(p,F) bind(C)
           import; implicit none;
           type(IGAPoint), intent(in) :: p
           real(kind=IGA_REAL_KIND), intent(out) :: F(p%dim,p%nsd)
         end subroutine IGAPoint_GradGeomMap
      end interface
      call IGAPoint_GradGeomMap(p,F)
    end function IGA_GradGeomMap

    function IGA_InvGradGeomMap(p) result (G)
      implicit none
      type(IGAPoint), intent(in) :: p
      real(kind=IGA_REAL_KIND)   :: G(p%nsd,p%dim)
      interface
         subroutine IGAPoint_InvGradGeomMap(p,G) bind(C)
           import; implicit none;
           type(IGAPoint), intent(in) :: p
           real(kind=IGA_REAL_KIND), intent(out) :: G(p%nsd,p%dim)
         end subroutine IGAPoint_InvGradGeomMap
      end interface
      call IGAPoint_InvGradGeomMap(p,G)
    end function IGA_InvGradGeomMap

    function IGA_Normal(p) result (N)
      use ISO_C_BINDING, only: c2f => C_F_POINTER
      implicit none
      type(IGAPoint), intent(in) :: p
      real(kind=IGA_REAL_KIND), pointer :: N(:)
      call c2f(p%normal,N,(/p%dim/))
    end function IGA_Normal

    function IGA_AtBoundary(p,axis,side) result (atboundary)
      implicit none
      type(IGAPoint), intent(in) :: p
      integer(kind=IGA_INTEGER_KIND), intent(out), optional :: axis
      integer(kind=IGA_INTEGER_KIND), intent(out), optional :: side
      integer(kind=IGA_INTEGER_KIND), parameter :: two = 2
      logical :: atboundary
      atboundary = (p%atboundary /= 0)
      if (atboundary) then
         if (present(axis)) axis = p%boundary_id/two
         if (present(side)) side = mod(p%boundary_id,two)
      else
         if (present(axis)) axis = -1
         if (present(side)) side = -1
      end if
    end function IGA_AtBoundary

    function IGA_Basis0(p) result(N)
      use ISO_C_BINDING, only: c2f => C_F_POINTER
      implicit none
      type(IGAPoint), intent(in) :: p
      real(kind=IGA_REAL_KIND), pointer :: N(:)
      call c2f(p%basis(0),N,(/p%nen/))
    end function IGA_Basis0

    function IGA_Basis1(p) result(N)
      use ISO_C_BINDING, only: c2f => C_F_POINTER
      implicit none
      type(IGAPoint), intent(in) :: p
      real(kind=IGA_REAL_KIND), pointer :: N(:,:)
      call c2f(p%basis(1),N,(/p%dim,p%nen/))
    end function IGA_Basis1

    function IGA_Basis2(p) result(N)
      use ISO_C_BINDING, only: c2f => C_F_POINTER
      implicit none
      type(IGAPoint), intent(in) :: p
      real(kind=IGA_REAL_KIND), pointer :: N(:,:,:)
      call c2f(p%basis(2),N,(/p%dim,p%dim,p%nen/))
    end function IGA_Basis2

    function IGA_Basis3(p) result(N)
      use ISO_C_BINDING, only: c2f => C_F_POINTER
      implicit none
      type(IGAPoint), intent(in) :: p
      real(kind=IGA_REAL_KIND), pointer :: N(:,:,:,:)
      call c2f(p%basis(3),N,(/p%dim,p%dim,p%dim,p%nen/))
    end function IGA_Basis3

    function IGA_Basis4(p) result(N)
      use ISO_C_BINDING, only: c2f => C_F_POINTER
      implicit none
      type(IGAPoint), intent(in) :: p
      real(kind=IGA_REAL_KIND), pointer :: N(:,:,:,:,:)
      call c2f(p%basis(4),N,(/p%dim,p%dim,p%dim,p%dim,p%nen/))
    end function IGA_Basis4

    function IGA_Shape0(p) result(N)
      use ISO_C_BINDING, only: c2f => C_F_POINTER
      implicit none
      type(IGAPoint), intent(in) :: p
      real(kind=IGA_REAL_KIND), pointer :: N(:)
      call c2f(p%shape(0),N,(/p%nen/))
    end function IGA_Shape0

    function IGA_Shape1(p) result(N)
      use ISO_C_BINDING, only: c2f => C_F_POINTER
      implicit none
      type(IGAPoint), intent(in) :: p
      real(kind=IGA_REAL_KIND), pointer :: N(:,:)
      call c2f(p%shape(1),N,(/p%dim,p%nen/))
    end function IGA_Shape1

    function IGA_Shape2(p) result(N)
      use ISO_C_BINDING, only: c2f => C_F_POINTER
      implicit none
      type(IGAPoint), intent(in) :: p
      real(kind=IGA_REAL_KIND), pointer :: N(:,:,:)
      call c2f(p%shape(2),N,(/p%dim,p%dim,p%nen/))
    end function IGA_Shape2

    function IGA_Shape3(p) result(N)
      use ISO_C_BINDING, only: c2f => C_F_POINTER
      implicit none
      type(IGAPoint), intent(in) :: p
      real(kind=IGA_REAL_KIND), pointer :: N(:,:,:,:)
      call c2f(p%shape(3),N,(/p%dim,p%dim,p%dim,p%nen/))
    end function IGA_Shape3

    function IGA_Shape4(p) result(N)
      use ISO_C_BINDING, only: c2f => C_F_POINTER
      implicit none
      type(IGAPoint), intent(in) :: p
      real(kind=IGA_REAL_KIND), pointer :: N(:,:,:,:,:)
      call c2f(p%shape(4),N,(/p%dim,p%dim,p%dim,p%dim,p%nen/))
    end function IGA_Shape4

#define DIM size(N,1)
#define DOF size(U,1)

    function IGA_Shape_Der0_S(N,U) result (V)
      implicit none
      real   (kind=IGA_REAL_KIND  ), intent(in) :: N(:) ! nen
      scalar (kind=IGA_SCALAR_KIND), intent(in) :: U(:) ! nen
      scalar (kind=IGA_SCALAR_KIND)  :: V
      ! V = dot_product(N,U)
      integer a
      V = 0
      do a = 1, size(U,1) ! nen
         V = V + N(a) * U(a)
      end do
    end function IGA_Shape_Der0_S

    function IGA_Shape_Der0_V(N,U) result (V)
      implicit none
      real   (kind=IGA_REAL_KIND  ), intent(in) :: N(:)   ! nen
      scalar (kind=IGA_SCALAR_KIND), intent(in) :: U(:,:) ! dof,nen
      scalar (kind=IGA_SCALAR_KIND)  :: V(DOF)            ! dof
      ! V = MATMUL(N,transpose(U))
      integer a
      V = 0
      do a = 1, size(U,2) ! nen
         V = V + N(a) * U(:,a)
      end do
    end function IGA_Shape_Der0_V

    function IGA_Shape_Der1_S(N,U) result (V)
      implicit none
      real   (kind=IGA_REAL_KIND  ), intent(in) :: N(:,:) ! dim,nen
      scalar (kind=IGA_SCALAR_KIND), intent(in) :: U(:)   ! nen
      scalar (kind=IGA_SCALAR_KIND)  :: V(DIM)            ! dim
      ! V = MATMUL(N,U)
      integer a
      V = 0
      do a = 1, size(U,1) ! nen
         V(:) = V(:) + N(:,a) * U(a)
      end do
    end function IGA_Shape_Der1_S

    function IGA_Shape_Der1_V(N,U) result (V)
      implicit none
      real   (kind=IGA_REAL_KIND  ), intent(in) :: N(:,:) ! dim,nen
      scalar (kind=IGA_SCALAR_KIND), intent(in) :: U(:,:) ! dof,nen
      scalar (kind=IGA_SCALAR_KIND)  :: V(DIM,DOF)        ! dim,dof
      ! V = MATMUL(N,transpose(U))
      integer a, c
      V = 0
      do a = 1, size(U,2) ! nen
         do c = 1, size(U,1) ! dof
            V(:,c) = V(:,c) + N(:,a) * U(c,a)
         end do
      end do
    end function IGA_Shape_Der1_V

    function IGA_Shape_Der2_S(N,U) result (V)
      implicit none
      real   (kind=IGA_REAL_KIND  ), intent(in) :: N(:,:,:) ! dim,dim,nen
      scalar (kind=IGA_SCALAR_KIND), intent(in) :: U(:)     ! nen
      scalar (kind=IGA_SCALAR_KIND)  :: V(DIM,DIM)          ! dim,dim
      integer a
      V = 0
      do a = 1, size(U,1) ! nen
         V(:,:) = V(:,:) + N(:,:,a) * U(a)
      end do
    end function IGA_Shape_Der2_S

    function IGA_Shape_Der2_V(N,U) result (V)
      implicit none
      real   (kind=IGA_REAL_KIND  ), intent(in) :: N(:,:,:) ! dim,dim,nen
      scalar (kind=IGA_SCALAR_KIND), intent(in) :: U(:,:)   ! dof,nen
      scalar (kind=IGA_SCALAR_KIND)  :: V(DIM,DIM,DOF)      ! dim,dim,dof
      integer a, c
      V = 0
      do a = 1, size(U,2) ! nen
         do c = 1, size(U,1) ! dof
            V(:,:,c) = V(:,:,c) + N(:,:,a) * U(c,a)
         end do
      end do
    end function IGA_Shape_Der2_V

    function IGA_Shape_Der3_S(N,U) result (V)
      implicit none
      real   (kind=IGA_REAL_KIND  ), intent(in) :: N(:,:,:,:) ! dim,dim,dim,nen
      scalar (kind=IGA_SCALAR_KIND), intent(in) :: U(:)       ! nen
      scalar (kind=IGA_SCALAR_KIND)  :: V(DIM,DIM,DIM)        ! dim,dim,dim
      integer a
      V = 0
      do a = 1, size(U,1) ! nen
         V(:,:,:) = V(:,:,:) + N(:,:,:,a) * U(a)
      end do
    end function IGA_Shape_Der3_S

    function IGA_Shape_Der3_V(N,U) result (V)
      implicit none
      real   (kind=IGA_REAL_KIND  ), intent(in) :: N(:,:,:,:) ! dim,dim,dim,nen
      scalar (kind=IGA_SCALAR_KIND), intent(in) :: U(:,:)     ! dof,nen
      scalar (kind=IGA_SCALAR_KIND)  :: V(DIM,DIM,DIM,DOF)    ! dim,dim,dim,dof
      integer a, c
      V = 0
      do a = 1, size(U,2) ! nen
         do c = 1, size(U,1) ! dof
            V(:,:,:,c) = V(:,:,:,c) + N(:,:,:,a) * U(c,a)
         end do
      end do
    end function IGA_Shape_Der3_V

    function IGA_Shape_Der4_S(N,U) result (V)
      implicit none
      real   (kind=IGA_REAL_KIND  ), intent(in) :: N(:,:,:,:,:) ! dim,dim,dim,dim,nen
      scalar (kind=IGA_SCALAR_KIND), intent(in) :: U(:)         ! nen
      scalar (kind=IGA_SCALAR_KIND)  :: V(DIM,DIM,DIM,DIM)      ! dim,dim,dim,dim
      integer a
      V = 0
      do a = 1, size(U,1) ! nen
         V(:,:,:,:) = V(:,:,:,:) + N(:,:,:,:,a) * U(a)
      end do
    end function IGA_Shape_Der4_S

    function IGA_Shape_Der4_V(N,U) result (V)
      implicit none
      real   (kind=IGA_REAL_KIND  ), intent(in) :: N(:,:,:,:,:) ! dim,dim,dim,dim,nen
      scalar (kind=IGA_SCALAR_KIND), intent(in) :: U(:,:)       ! dof,nen
      scalar (kind=IGA_SCALAR_KIND)  :: V(DIM,DIM,DIM,DIM,DOF)  ! dim,dim,dim,dim,dof
      integer a, c
      V = 0
      do a = 1, size(U,2) ! nen
         do c = 1, size(U,1) ! dof
            V(:,:,:,:,c) = V(:,:,:,:,c) + N(:,:,:,:,a) * U(c,a)
         end do
      end do
    end function IGA_Shape_Der4_V

    function IGA_Shape_Div(N,U) result (V)
      implicit none
      real   (kind=IGA_REAL_KIND  ), intent(in) :: N(:,:) ! dim,nen
      scalar (kind=IGA_SCALAR_KIND), intent(in) :: U(:,:) ! dim,nen
      scalar (kind=IGA_SCALAR_KIND)  :: V
      integer a, i
      V = 0
      do a = 1, size(U,2) ! nen
         do i = 1, size(N,1) ! dim
            V = V + N(i,a) * U(i,a)
         end do
      end do
    end function IGA_Shape_Div

    function IGA_Shape_Del2_S(N,U) result (V)
      implicit none
      real   (kind=IGA_REAL_KIND  ), intent(in) :: N(:,:,:) ! dim,dim,nen
      scalar (kind=IGA_SCALAR_KIND), intent(in) :: U(:)     ! nen
      scalar (kind=IGA_SCALAR_KIND)  :: V
      integer a, i
      V = 0
      do a = 1, size(U,1) ! nen
         do i = 1, size(N,1) ! dim
            V = V + N(i,i,a) * U(a)
         end do
      end do
    end function IGA_Shape_Del2_S

    function IGA_Shape_Del2_V(N,U) result (V)
      implicit none
      real   (kind=IGA_REAL_KIND  ), intent(in) :: N(:,:,:) ! dim,dim,nen
      scalar (kind=IGA_SCALAR_KIND), intent(in) :: U(:,:)   ! dof,nen
      scalar (kind=IGA_SCALAR_KIND)  :: V(DOF)
      integer a, c, i
      V = 0
      do a = 1, size(U,2) ! nen
         do c = 1, size(U,1) ! dof
            do i = 1, size(N,1) ! dim
               V(c) = V(c) + N(i,i,a) * U(c,a)
            end do
         end do
      end do
    end function IGA_Shape_Del2_V

#undef DIM
#undef DOF

#define DIM p%dim
#define DOF size(U,1)

    function IGA_Point_Der0_S(p,U) result(V)
      implicit none
      type(IGAPoint), intent(in) :: p
      scalar (kind=IGA_SCALAR_KIND), intent(in) :: U(:) ! nen
      scalar (kind=IGA_SCALAR_KIND)  :: V               !
      V = IGA_Eval(IGA_Shape0(p),U)
    end function IGA_Point_Der0_S

    function IGA_Point_Der0_V(p,U) result(V)
      implicit none
      type(IGAPoint), intent(in) :: p
      scalar (kind=IGA_SCALAR_KIND), intent(in) :: U(:,:) ! dof,nen
      scalar (kind=IGA_SCALAR_KIND)  :: V(DOF)            ! dim,dof
      V = IGA_Eval(IGA_Shape0(p),U)
    end function IGA_Point_Der0_V

    function IGA_Point_Der1_S(p,U) result(V)
      implicit none
      type(IGAPoint), intent(in) :: p
      scalar (kind=IGA_SCALAR_KIND), intent(in) :: U(:) ! nen
      scalar (kind=IGA_SCALAR_KIND)  :: V(DIM)          ! dim
      V = IGA_Eval(IGA_Shape1(p),U)
    end function IGA_Point_Der1_S

    function IGA_Point_Der1_V(p,U) result(V)
      implicit none
      type(IGAPoint), intent(in) :: p
      scalar (kind=IGA_SCALAR_KIND), intent(in) :: U(:,:) ! dof,nen
      scalar (kind=IGA_SCALAR_KIND)  :: V(DIM,DOF)        ! dim,dof
      V = IGA_Eval(IGA_Shape1(p),U)
    end function IGA_Point_Der1_V

    function IGA_Point_Der2_S(p,U) result(V)
      implicit none
      type(IGAPoint), intent(in) :: p
      scalar (kind=IGA_SCALAR_KIND), intent(in) :: U(:) ! nen
      scalar (kind=IGA_SCALAR_KIND)  :: V(DIM,DIM)      ! dim,dim
      V = IGA_Eval(IGA_Shape2(p),U)
    end function IGA_Point_Der2_S

    function IGA_Point_Der2_V(p,U) result(V)
      implicit none
      type(IGAPoint), intent(in) :: p
      scalar (kind=IGA_SCALAR_KIND), intent(in) :: U(:,:) ! dof,nen
      scalar (kind=IGA_SCALAR_KIND)  :: V(DIM,DIM,DOF)    ! dim,dim,dof
      V = IGA_Eval(IGA_Shape2(p),U)
    end function IGA_Point_Der2_V

    function IGA_Point_Der3_S(p,U) result(V)
      implicit none
      type(IGAPoint), intent(in) :: p
      scalar (kind=IGA_SCALAR_KIND), intent(in) :: U(:) ! nen
      scalar (kind=IGA_SCALAR_KIND)  :: V(DIM,DIM,DIM)  ! dim,dim,dim
      V = IGA_Eval(IGA_Shape3(p),U)
    end function IGA_Point_Der3_S

    function IGA_Point_Der3_V(p,U) result(V)
      implicit none
      type(IGAPoint), intent(in) :: p
      scalar (kind=IGA_SCALAR_KIND), intent(in) :: U(:,:)  ! dof,nen
      scalar (kind=IGA_SCALAR_KIND)  :: V(DIM,DIM,DIM,DOF) ! dim,dim,dim,dof
      V = IGA_Eval(IGA_Shape3(p),U)
    end function IGA_Point_Der3_V

    function IGA_Point_Der4_S(p,U) result(V)
      implicit none
      type(IGAPoint), intent(in) :: p
      scalar (kind=IGA_SCALAR_KIND), intent(in) :: U(:)    ! nen
      scalar (kind=IGA_SCALAR_KIND)  :: V(DIM,DIM,DIM,DIM) ! dim,dim,dim,dim
      V = IGA_Eval(IGA_Shape4(p),U)
    end function IGA_Point_Der4_S

    function IGA_Point_Der4_V(p,U) result(V)
      implicit none
      type(IGAPoint), intent(in) :: p
      scalar (kind=IGA_SCALAR_KIND), intent(in) :: U(:,:)      ! dof,nen
      scalar (kind=IGA_SCALAR_KIND)  :: V(DIM,DIM,DIM,DIM,DOF) ! dim,dim,dim,dim,dof
      V = IGA_Eval(IGA_Shape4(p),U)
    end function IGA_Point_Der4_V

    function IGA_Point_Div(p,U) result (V)
      implicit none
      type(IGAPoint), intent(in) :: p
      scalar (kind=IGA_SCALAR_KIND), intent(in) :: U(:,:) ! dim,nen
      scalar (kind=IGA_SCALAR_KIND)  :: V(DIM)            ! dim
      V = IGA_Shape_Div(IGA_Shape1(p),U)
    end function IGA_Point_Div

    function IGA_Point_Del2_S(p,U) result (V)
      implicit none
      type(IGAPoint), intent(in) :: p
      scalar (kind=IGA_SCALAR_KIND), intent(in) :: U(:) ! nen
      scalar (kind=IGA_SCALAR_KIND)  :: V
      V = IGA_Shape_Del2_S(IGA_Shape2(p),U)
    end function IGA_Point_Del2_S

    function IGA_Point_Del2_V(p,U) result (V)
      implicit none
      type(IGAPoint), intent(in) :: p
      scalar (kind=IGA_SCALAR_KIND), intent(in) :: U(:,:) ! dof,nen
      scalar (kind=IGA_SCALAR_KIND)  :: V(DOF)
      V = IGA_Shape_Del2_V(IGA_Shape2(p),U)
    end function IGA_Point_Del2_V

#undef DIM
#undef DOF

end module PetIGA
