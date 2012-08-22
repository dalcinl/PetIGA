! -*- f90 -*-

subroutine IGA_DersBasisFuns(i,uu,p,d,U,N) &
  bind(C, name="IGA_DersBasisFuns")
  use PetIGA
  implicit none
  interface
     pure subroutine DersBasisFuns(i,uu,p,d,U,ders)
       use PetIGA
       integer(kind=IGA_INTEGER_KIND), intent(in)  :: i, p, d
       real   (kind=IGA_REAL_KIND   ), intent(in)  :: uu, U(0:i+p)
       real   (kind=IGA_REAL_KIND   ), intent(out) :: ders(0:p,0:d)
     end subroutine DersBasisFuns
  end interface
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: i, p, d
  real   (kind=IGA_REAL_KIND   ), intent(in),value :: uu
  real   (kind=IGA_REAL_KIND   ), intent(in)       :: U(0:i+p)
  real   (kind=IGA_REAL_KIND   ), intent(out)      :: N(0:d,0:p)
  real   (kind=IGA_REAL_KIND   )  :: ders(0:p,0:d)
  call DersBasisFuns(i,uu,p,d,U,ders)
  N = transpose(ders)
end subroutine IGA_DersBasisFuns

pure subroutine DersBasisFuns(i,uu,p,n,U,ders)
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
end subroutine DersBasisFuns
