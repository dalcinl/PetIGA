! -*- f90 -*-

subroutine DersBasisFuns(i,uu,p,n,U,ders)
  use ISO_C_BINDING, only: C_INT, C_LONG
  use ISO_C_BINDING, only: C_FLOAT, C_DOUBLE
  implicit none
  integer(kind=C_INT),    intent(in) :: i, p, n
  real   (kind=C_DOUBLE), intent(in) :: uu, U(0:i+p)
  real   (kind=C_DOUBLE), intent(out):: ders(0:p,0:n)
  integer(kind=C_INT)    :: j, k, r, s1, s2, rk, pk, j1, j2
  real   (kind=C_DOUBLE) :: saved, temp, d
  real   (kind=C_DOUBLE) :: left(p), right(p)
  real   (kind=C_DOUBLE) :: ndu(0:p,0:p), a(0:1,0:p)
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

subroutine IGA_DersBasisFuns(i,uu,p,d,U,N) &
  bind(C, name="IGA_DersBasisFuns")
  use ISO_C_BINDING, only: C_INT, C_LONG
  use ISO_C_BINDING, only: C_FLOAT, C_DOUBLE
  implicit none
  interface
     subroutine DersBasisFuns(i,uu,p,d,U,ders)
       use ISO_C_BINDING, only: C_INT, C_LONG
       use ISO_C_BINDING, only: C_FLOAT, C_DOUBLE
       integer(kind=C_INT),    intent(in) :: i, p, d
       real   (kind=C_DOUBLE), intent(in) :: uu, U(0:i+p)
       real   (kind=C_DOUBLE), intent(out):: ders(0:p,0:d)
     end subroutine DersBasisFuns
  end interface
  integer(kind=C_INT),    intent(in),value :: i, p, d
  real   (kind=C_DOUBLE), intent(in),value :: uu
  real   (kind=C_DOUBLE), intent(in)       :: U(0:i+p)
  real   (kind=C_DOUBLE), intent(out)      :: N(0:d,0:p)
  real   (kind=C_DOUBLE) ders(0:p,0:d)
  call DersBasisFuns(i,uu,p,d,U,ders)
  N = transpose(ders)
end subroutine IGA_DersBasisFuns
