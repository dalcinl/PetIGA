! -*- f90 -*-

#include "petscconf.h"
#if defined(PETSC_USE_COMPLEX)
#define scalar complex
#else
#define scalar real
#endif

pure subroutine IGA_GeometryMap(&
     order,dim,nsd,             &
     nqp,nen,X,                 &
     M0,M1,M2,M3,M4,            &
     X0,X1,X2,X3,X4)            &
  bind(C, name="IGA_GeometryMap")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: order
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: dim
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: nsd
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: nqp
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: nen
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: X(        nsd,nen)
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: M0(dim**0*nen,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: M1(dim**1*nen,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: M2(dim**2*nen,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: M3(dim**3*nen,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: M4(dim**4*nen,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: X0(dim**0*nsd,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: X1(dim**1*nsd,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: X2(dim**2*nsd,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: X3(dim**3*nsd,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: X4(dim**4*nsd,nqp)
  integer(kind=IGA_INTEGER_KIND)  :: q
  do q=1,nqp
     call GeometryMap(&
          order,nen,X,&
          M0(:,q),M1(:,q),M2(:,q),M3(:,q),M4(:,q),&
          X0(:,q),X1(:,q),X2(:,q),X3(:,q),X4(:,q))
  end do
contains
include 'petigamapgeo.f90.in'
end subroutine IGA_GeometryMap

subroutine IGA_GetNormal(dim,axis,side,F,dS,N) &
  bind(C, name="IGA_GetNormal")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: dim
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: axis,side
  real   (kind=IGA_REAL_KIND   ), intent(in)       :: F(dim,dim)
  real   (kind=IGA_REAL_KIND   ), intent(out)      :: dS
  real   (kind=IGA_REAL_KIND   ), intent(out)      :: N(dim)
  select case (dim)
  case (3); dS = normal3d(axis,F,N)
  case (2); dS = normal2d(axis,F,N)
  case (1); dS = 1; N(1) = 1
  end select
  if (side == 0) N = -N
contains
function normal2d(axis,F,N) result(dS)
  implicit none
  integer(kind=IGA_INTEGER_KIND), intent(in) :: axis
  real(kind=IGA_REAL_KIND), intent(in)  :: F(2,2)
  real(kind=IGA_REAL_KIND), intent(out) :: N(2)
  real(kind=IGA_REAL_KIND) :: dS
  real(kind=IGA_REAL_KIND) :: t(2)
  select case (axis)
  case (0); t = +F(2,:)
  case (1); t = -F(1,:)
  case default; t = 0
  end select
  ! n_i = eps_ij n_j
  N(1) = +t(2)
  N(2) = -t(1)
  dS = sqrt(sum(N*N))
  N = N/dS
end function normal2d
function normal3d(axis,F,N) result(dS)
  implicit none
  integer(kind=IGA_INTEGER_KIND), intent(in) :: axis
  real(kind=IGA_REAL_KIND), intent(in)  :: F(3,3)
  real(kind=IGA_REAL_KIND), intent(out) :: N(3)
  real(kind=IGA_REAL_KIND) :: dS
  real(kind=IGA_REAL_KIND) :: s(3), t(3)
  select case (axis)
  case (0); s = F(2,:); t = F(3,:)
  case (1); s = F(3,:); t = F(1,:)
  case (2); s = F(1,:); t = F(2,:)
  case default;  s = 0; t = 0
  end select
  ! n_i = eps_ijk s_j t_k
  N(1) = s(2) * t(3) - s(3) * t(2)
  N(2) = s(3) * t(1) - s(1) * t(3)
  N(3) = s(1) * t(2) - s(2) * t(1)
  dS = sqrt(sum(N*N))
  N = N/dS
end function normal3d
end subroutine IGA_GetNormal


subroutine IGA_GetGeomMap(nen,nsd,N,C,X) &
  bind(C, name="IGA_GetGeomMap")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: nen,nsd
  real   (kind=IGA_REAL_KIND   ), intent(in)       :: N(nen)
  real   (kind=IGA_REAL_KIND   ), intent(in)       :: C(nsd,nen)
  real   (kind=IGA_REAL_KIND   ), intent(out)      :: X(nsd)
  X = matmul(C,N)
end subroutine IGA_GetGeomMap

subroutine IGA_GetGradGeomMap(nen,nsd,dim,N,C,F) &
  bind(C, name="IGA_GetGradGeomMap")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: nen,nsd,dim
  real   (kind=IGA_REAL_KIND   ), intent(in)       :: N(dim,nen)
  real   (kind=IGA_REAL_KIND   ), intent(in)       :: C(nsd,nen)
  real   (kind=IGA_REAL_KIND   ), intent(out)      :: F(dim,nsd)
  F = matmul(N,transpose(C))
end subroutine IGA_GetGradGeomMap

subroutine IGA_GetInvGradGeomMap(nen,nsd,dim,N,C,G) &
  bind(C, name="IGA_GetInvGradGeomMap")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: nen,nsd,dim
  real   (kind=IGA_REAL_KIND   ), intent(in)       :: N(dim,nen)
  real   (kind=IGA_REAL_KIND   ), intent(in)       :: C(nsd,nen)
  real   (kind=IGA_REAL_KIND   ), intent(out)      :: G(nsd,dim)
  real   (kind=IGA_REAL_KIND   )  :: F(nsd,dim)
  real   (kind=IGA_REAL_KIND   )  :: M(dim,dim), detM, invM(dim,dim)
  F = matmul(C,transpose(N))
  M = matmul(transpose(F),F)
  detM = Determinant(dim,M)
  call Inverse(dim,detM,M,invM)
  G = transpose(matmul(invM,transpose(F)))
contains
include 'petigadet.f90.in'
include 'petigainv.f90.in'
end subroutine IGA_GetInvGradGeomMap


subroutine IGA_EvaluateReal(nen,dof,dim,N,U,V) &
  bind(C, name="IGA_EvaluateReal")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: nen,dof,dim
  real   (kind=IGA_REAL_KIND   ), intent(in)       :: N(dim,nen)
  real   (kind=IGA_REAL_KIND   ), intent(in)       :: U(dof,nen)
  real   (kind=IGA_REAL_KIND   ), intent(out)      :: V(dim,dof)
  integer(kind=IGA_INTEGER_KIND)  :: a, i
  ! V = MATMUL(N,transpose(U))
  V = 0
  do a = 1, nen
     do i = 1, dof
        V(:,i) = V(:,i) + N(:,a) * U(i,a)
     end do
  end do
end subroutine IGA_EvaluateReal

subroutine IGA_EvaluateScalar(nen,dof,dim,N,U,V) &
  bind(C, name="IGA_EvaluateScalar")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: nen,dof,dim
  real   (kind=IGA_REAL_KIND   ), intent(in)       :: N(dim,nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(in)       :: U(dof,nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(out)      :: V(dim,dof)
  integer(kind=IGA_INTEGER_KIND)  :: a, i
  ! V = MATMUL(N,transpose(U))
  V = 0
  do a = 1, nen
     do i = 1, dof
        V(:,i) = V(:,i) + N(:,a) * U(i,a)
     end do
  end do
end subroutine IGA_EvaluateScalar


subroutine IGA_GetValue(nen,dof,N,U,V) &
  bind(C, name="IGA_GetValue")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: nen,dof
  real   (kind=IGA_REAL_KIND   ), intent(in)       :: N(nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(in)       :: U(dof,nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(out)      :: V(dof)
  integer(kind=IGA_INTEGER_KIND)  :: a
  ! V = MATMUL(N,transpose(U))
  V = 0
  do a = 1, nen
     V = V + N(a) * U(:,a)
  end do
end subroutine IGA_GetValue

subroutine IGA_GetGrad(nen,dof,dim,N,U,V) &
  bind(C, name="IGA_GetGrad")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: nen,dof,dim
  real   (kind=IGA_REAL_KIND   ), intent(in)       :: N(dim,nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(in)       :: U(dof,nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(out)      :: V(dim,dof)
  integer(kind=IGA_INTEGER_KIND)  :: a, i
  ! V = MATMUL(N,transpose(U))
  V = 0
  do a = 1, nen
     do i = 1, dof
        V(:,i) = V(:,i) + N(:,a) * U(i,a)
     end do
  end do
end subroutine IGA_GetGrad

subroutine IGA_GetHess(nen,dof,dim,N,U,V) &
  bind(C, name="IGA_GetHess")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: nen,dof,dim
  real   (kind=IGA_REAL_KIND   ), intent(in)       :: N(dim*dim,nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(in)       :: U(dof,nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(out)      :: V(dim*dim,dof)
  integer(kind=IGA_INTEGER_KIND)  :: a, i
  ! V = MATMUL(N,transpose(U))
  V = 0
  do a = 1, nen
     do i = 1, dof
        V(:,i) = V(:,i) + N(:,a) * U(i,a)
     end do
  end do
end subroutine IGA_GetHess

subroutine IGA_GetDel2(nen,dof,dim,N,U,V) &
  bind(C, name="IGA_GetDel2")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: nen,dof,dim
  real   (kind=IGA_REAL_KIND   ), intent(in)       :: N(dim,dim,nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(in)       :: U(dof,nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(out)      :: V(dof)
  integer(kind=IGA_INTEGER_KIND)  :: a, c, i
  V = 0
  do a = 1, nen
     do c = 1, dof
        do i = 1, dim
           V(c) = V(c) + N(i,i,a) * U(c,a)
        end do
     end do
  end do
end subroutine IGA_GetDel2

subroutine IGA_GetDer3(nen,dof,dim,N,U,V) &
     bind(C, name="IGA_GetDer3")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: nen,dof,dim
  real   (kind=IGA_REAL_KIND   ), intent(in)       :: N(dim*dim*dim,nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(in)       :: U(dof,nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(out)      :: V(dim*dim*dim,dof)
  integer(kind=IGA_INTEGER_KIND)  :: a, i
  ! V = MATMUL(N,transpose(U))
  V = 0
  do a = 1, nen
     do i = 1, dof
        V(:,i) = V(:,i) + N(:,a) * U(i,a)
     end do
  end do
end subroutine IGA_GetDer3

subroutine IGA_GetDer4(nen,dof,dim,N,U,V) &
     bind(C, name="IGA_GetDer4")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: nen,dof,dim
  real   (kind=IGA_REAL_KIND   ), intent(in)       :: N(dim*dim*dim*dim,nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(in)       :: U(dof,nen)
  scalar (kind=IGA_SCALAR_KIND ), intent(out)      :: V(dim*dim*dim*dim,dof)
  integer(kind=IGA_INTEGER_KIND)  :: a, i
  ! V = MATMUL(N,transpose(U))
  V = 0
  do a = 1, nen
     do i = 1, dof
        V(:,i) = V(:,i) + N(:,a) * U(i,a)
     end do
  end do
end subroutine IGA_GetDer4
