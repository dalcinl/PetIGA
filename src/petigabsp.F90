! -*- f90 -*-

subroutine IGA_Basis_BSpline(k,uu,p,d,U,B) &
  bind(C, name="IGA_Basis_BSpline")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: k, p, d
  real   (kind=IGA_REAL_KIND   ), intent(in),value :: uu
  real   (kind=IGA_REAL_KIND   ), intent(in)       :: U(0:k+p)
  real   (kind=IGA_REAL_KIND   ), intent(out)      :: B(0:4,0:p)
  real   (kind=IGA_REAL_KIND   )  :: ders(0:p,0:d)
  call BSplineBasisFunsDers(k,uu,p,d,U,ders)
  B = 0; B(0:d,:) = transpose(ders)
contains
include 'petigabsb.f90.in'
end subroutine IGA_Basis_BSpline


subroutine IGA_Basis_Lagrange(k,uu,p,d,U,B) &
  bind(C, name="IGA_Basis_Lagrange")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: k, p, d
  real   (kind=IGA_REAL_KIND   ), intent(in),value :: uu
  real   (kind=IGA_REAL_KIND   ), intent(in)       :: U(0:k+1)
  real   (kind=IGA_REAL_KIND   ), intent(out)      :: B(0:4,0:p)
  real   (kind=IGA_REAL_KIND   )  :: X(0:p)
  if (p == 0) then
     B(0,0) = 1
     return
  end if
  B = 0
  call NewtonCotesPoints(p+1,U(k),U(k+1),X)
  call LagrangeBasisFunsDers(uu,p,d,X,B)
contains
include 'petigalgb.f90.in'
pure subroutine NewtonCotesPoints(n,U0,U1,X)
  implicit none
  integer, parameter :: rk = IGA_REAL_KIND
  integer(kind=IGA_INTEGER_KIND), intent(in)  :: n
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: U0, U1
  real   (kind=IGA_REAL_KIND   ), intent(out) :: X(0:n-1)
  integer(kind=IGA_INTEGER_KIND)  :: i
  do i = 0, n-1
     X(i) = U0 + (U1-U0) * real(i,rk)/real(n-1,rk)
  end do
end subroutine NewtonCotesPoints
end subroutine IGA_Basis_Lagrange


subroutine IGA_Basis_Spectral(k,uu,p,d,U,B) &
  bind(C, name="IGA_Basis_Spectral")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: k, p, d
  real   (kind=IGA_REAL_KIND   ), intent(in),value :: uu
  real   (kind=IGA_REAL_KIND   ), intent(in)       :: U(0:k+1)
  real   (kind=IGA_REAL_KIND   ), intent(out)      :: B(0:4,0:p)
  real   (kind=IGA_REAL_KIND   )  :: X(0:p)
  if (p == 0) then
     B(0,0) = 1
     return
  end if
  B = 0
  call GaussLobattoPoints(p+1,U(k),U(k+1),X)
  call LagrangeBasisFunsDers(uu,p,d,X,B)
contains
include 'petigaglp.f90.in'
include 'petigalgb.f90.in'
end subroutine IGA_Basis_Spectral


subroutine IGA_LobattoPoints(n,x0,x1,X) &
  bind(C, name="IGA_LobattoPoints")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), intent(in),value  :: n
  real   (kind=IGA_REAL_KIND   ), intent(in),value  :: x0, x1
  real   (kind=IGA_REAL_KIND   ), intent(out)       :: X(0:n-1)
  call GaussLobattoPoints(n,x0,x1,X)
contains
include 'petigaglp.f90.in'
end subroutine IGA_LobattoPoints
