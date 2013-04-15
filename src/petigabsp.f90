! -*- f90 -*-

subroutine IGA_Basis_BSpline(k,uu,p,d,U,B) &
  bind(C, name="IGA_Basis_BSpline")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: k, p, d
  real   (kind=IGA_REAL_KIND   ), intent(in),value :: uu
  real   (kind=IGA_REAL_KIND   ), intent(in)       :: U(0:k+p)
  real   (kind=IGA_REAL_KIND   ), intent(out)      :: B(0:d,0:p)
  real   (kind=IGA_REAL_KIND   )  :: ders(0:p,0:d)
  call BasisFunsDers(k,uu,p,d,U,ders)
  B = transpose(ders)
contains
pure subroutine BasisFunsDers(i,uu,p,n,U,ders)
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), intent(in)  :: i, p, n
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: uu, U(0:i+p)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: ders(0:p,0:n)
  integer(kind=IGA_INTEGER_KIND)  :: j, k, r, s1, s2, rk, pk, j1, j2
  real   (kind=IGA_REAL_KIND   )  :: saved, temp, d
  real   (kind=IGA_REAL_KIND   )  :: left(p), right(p)
  real   (kind=IGA_REAL_KIND   )  :: ndu(0:p,0:p), a(0:1,0:p)
  ndu(0,0) = 1.0
  do j = 1, p
     left(j)  = uu - U(i+1-j)
     right(j) = U(i+j) - uu
     saved = 0.0
     do r = 0, j-1
        ndu(j,r) = right(r+1) + left(j-r)
        temp = ndu(r,j-1) / ndu(j,r)
        ndu(r,j) = saved + right(r+1) * temp
        saved = left(j-r) * temp
     end do
     ndu(j,j) = saved
  end do
  ders(:,0) = ndu(:,p)
  do r = 0, p
     s1 = 0; s2 = 1;
     a(0,0) = 1.0
     do k = 1, n
        d = 0.0
        rk = r-k; pk = p-k;
        if (r >= k) then
           a(s2,0) = a(s1,0) / ndu(pk+1,rk)
           d =  a(s2,0) * ndu(rk,pk)
        end if
        if (rk > -1) then
           j1 = 1
        else
           j1 = -rk
        end if
        if (r-1 <= pk) then
           j2 = k-1
        else
           j2 = p-r
        end if
        do j = j1, j2
           a(s2,j) = (a(s1,j) - a(s1,j-1)) / ndu(pk+1,rk+j)
           d =  d + a(s2,j) * ndu(rk+j,pk)
        end do
        if (r <= pk) then
           a(s2,k) = - a(s1,k-1) / ndu(pk+1,r)
           d =  d + a(s2,k) * ndu(r,pk)
        end if
        ders(r,k) = d
        j = s1; s1 = s2; s2 = j;
     end do
  end do
  r = p
  do k = 1, n
     ders(:,k) = ders(:,k) * r
     r = r * (p-k)
  end do
end subroutine BasisFunsDers
end subroutine IGA_Basis_BSpline


subroutine IGA_Basis_Lagrange(kk,uu,p,d,U,B) &
  bind(C, name="IGA_Basis_Lagrange")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: kk, p, d
  real   (kind=IGA_REAL_KIND   ), intent(in),value :: uu
  real   (kind=IGA_REAL_KIND   ), intent(in)       :: U(0:kk+p)
  real   (kind=IGA_REAL_KIND   ), intent(out)      :: B(0:d,0:p)
  integer(kind=IGA_INTEGER_KIND)  :: m, i, j, k, l
  real   (kind=IGA_REAL_KIND   )  :: Lp, Ls1, Ls2, Ls3
  real   (kind=IGA_REAL_KIND   )  :: X(0:p)

  do m = 0, p
     X(m) = U(kk) + m * (U(kk+1) - U(kk)) / p
  end do

  do m = 0, p
     Lp = 1.0
     do i = 0, p
        if (i == m) cycle
        Lp = Lp * (uu-X(i))/(X(m)-X(i))
     end do
     B(0,m) = Lp
  end do

  if (d < 1) return
  do m = 0, p
     Ls1 = 0.0
     do j = 0, p
        if (j == m) cycle
        Lp = 1.0
        do i = 0, p
           if (i == m .or. i == j) cycle
           Lp = Lp * (uu-X(i))/(X(m)-X(i))
        end do
        Ls1 = Ls1 + Lp/(X(m)-X(j))
     end do
     B(1,m) = Ls1
  end do

  if (d < 2) return
  do m = 0, p
     Ls2 = 0.0
     do k = 0, p
        if (k == m) cycle
        Ls1 = 0.0
        do j = 0, p
           if (j == m .or. j == k) cycle
           Lp = 1.0
           do i = 0, p
              if (i == m .or. i == k .or. i == j) cycle
              Lp = Lp * (uu-X(i))/(X(m)-X(i))
           end do
           Ls1 = Ls1 + Lp/(X(m)-X(j))
        end do
        Ls2 = Ls2 + Ls1/(X(m)-X(k))
     end do
     B(2,m) = Ls2
  end do

  if (d < 3) return
  do m = 0, p
     Ls3 = 0.0
     do l = 0, p
        if (l == m) cycle
        Ls2 = 0.0
        do k = 0, p
           if (k == m .or. k == l) cycle
           Ls1 = 0.0
           do j = 0, p
              if (j == m .or. j == l .or. j == k) cycle
              Lp = 1.0
              do i = 0, p
                 if (i == m .or. i == l .or. i == k .or. i == j) cycle
                 Lp = Lp * (uu-X(i))/(X(m)-X(i))
              end do
              Ls1 = Ls1 + Lp/(X(m)-X(j))
           end do
           Ls2 = Ls2 + Ls1/(X(m)-X(k))
        end do
        Ls3 = Ls3 + Ls2/(X(m)-X(l))
     end do
     B(3,m) = Ls3
  end do

end subroutine IGA_Basis_Lagrange


subroutine IGA_Basis_Hierarchical(kk,uu,p,d,U,B) &
  bind(C, name="IGA_Basis_Hierarchical")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: kk, p, d
  real   (kind=IGA_REAL_KIND   ), intent(in),value :: uu
  real   (kind=IGA_REAL_KIND   ), intent(in)       :: U(0:kk+p)
  real   (kind=IGA_REAL_KIND   ), intent(out)      :: B(0:d,0:p)
  integer(kind=IGA_INTEGER_KIND)  :: i, k
  real   (kind=IGA_REAL_KIND   )  :: J, x, Lp(0:p,0:d)
  real   (kind=IGA_REAL_KIND   ), parameter :: two = 2.0

  J = (U(kk+1)-U(kk))/2.0
  x = (uu-U(kk))/J - 1.0

  B(0,0) = (1.0-x)/2.0
  B(0,p) = (x+1.0)/2.0
  if (d > 0) then
     B(1,0) = -0.5
     B(1,p) = +0.5
  endif

  if (p > 1) then
     Lp(:,:) = 0.0
     Lp(0,0) = 1.0
     Lp(1,0) = x
     if (d > 0) then
        Lp(0,1) = 0.0
        Lp(1,1) = 1.0
     end if
     do i = 1, p-1
        Lp(i+1,0) = ((2*i+1)*x*Lp(i,0) - i*Lp(i-1,0))/(i+1)
        B(0,i) = (-Lp(i+1,0) + Lp(i-1,0))/sqrt(two*(2*(i+1)-1))
        do k = 1, d
           Lp(i+1,k) = (2*i+1)*Lp(i,k-1) + Lp(i-1,k)
           B(k,i) = (-Lp(i+1,k) + Lp(i-1,k))/sqrt(two*(2*(i+1)-1))
        end do
     end do
  end if

  do k = 1, d
     B(k,:) = B(k,:)/(J**k)
  end do

end subroutine IGA_Basis_Hierarchical
