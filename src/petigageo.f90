! -*- f90 -*-

subroutine IGA_ShapeFuns(&
     ord,                &
     dim,nen,nqp,        &
     geometry,X,         &
     M0,M1,M2,M3,        &
     N0,N1,N2,N3,        &
     DetJac,Jac)         &
  bind(C, name="IGA_ShapeFuns")
  use PetIGA
  implicit none
  integer(kind=IGA_INT ), intent(in),value :: ord
  integer(kind=IGA_INT ), intent(in),value :: dim
  integer(kind=IGA_INT ), intent(in),value :: nen, nqp
  integer(kind=IGA_INT ), intent(in),value :: geometry
  real   (kind=IGA_REAL), intent(in)    :: X(dim+1,nen)
  real   (kind=IGA_REAL), intent(in)    :: M0(       nen,nqp)
  real   (kind=IGA_REAL), intent(in)    :: M1(dim,   nen,nqp)
  real   (kind=IGA_REAL), intent(in)    :: M2(dim**2,nen,nqp)
  real   (kind=IGA_REAL), intent(in)    :: M3(dim**3,nen,nqp)
  real   (kind=IGA_REAL), intent(out)   :: N0(       nen,nqp)
  real   (kind=IGA_REAL), intent(out)   :: N1(dim,   nen,nqp)
  real   (kind=IGA_REAL), intent(out)   :: N2(dim**2,nen,nqp)
  real   (kind=IGA_REAL), intent(out)   :: N3(dim**3,nen,nqp)
  real   (kind=IGA_REAL), intent(inout) :: DetJac(       nqp)
  real   (kind=IGA_REAL), intent(inout) :: Jac(dim,dim,  nqp)

  integer(kind=IGA_INT )  :: q, i
  real   (kind=IGA_REAL)  :: qDJ, qJ(dim,dim)

  if (geometry /= 0) then
     do q=1,nqp
        call IsoparametricMapping(&
             ord,&
             dim,nen,X,&
             M0(:,q),M1(:,:,q),M2(:,:,q),M3(:,:,q),&
             N0(:,q),N1(:,:,q),N2(:,:,q),N3(:,:,q),&
             qDJ,qJ)
        detJac(q)  = detJac(q)  * qDJ
        Jac(:,:,q) = Jac(:,:,q) * qJ
     end do
  else
     N0 = M0; N1 = M1;
     N2 = M2; N3 = M3;
     detJac = 1
     Jac = 0
     forall(i=1:dim)
        Jac(i,i,:) = 1
     end forall
  end if

contains

pure subroutine IsoparametricMapping(&
     ord,&
     dim,nen,X,&
     N0,N1,N2,N3,&
     R0,R1,R2,R3,&
     DetJ,Jaco)
  implicit none
  integer(kind=IGA_INT ), intent(in)  :: ord
  integer(kind=IGA_INT ), intent(in)  :: dim, nen
  real   (kind=IGA_REAL), intent(in)  :: X(dim,nen)
  real   (kind=IGA_REAL), intent(in)  :: N0(            nen)
  real   (kind=IGA_REAL), intent(in)  :: N1(        dim,nen)
  real   (kind=IGA_REAL), intent(in)  :: N2(    dim,dim,nen)
  real   (kind=IGA_REAL), intent(in)  :: N3(dim,dim,dim,nen)
  real   (kind=IGA_REAL), intent(out) :: R0(            nen)
  real   (kind=IGA_REAL), intent(out) :: R1(        dim,nen)
  real   (kind=IGA_REAL), intent(out) :: R2(    dim,dim,nen)
  real   (kind=IGA_REAL), intent(out) :: R3(dim,dim,dim,nen)
  real   (kind=IGA_REAL), intent(out) :: DetJ, Jaco(dim,dim)

  integer(kind=IGA_INT ) :: idx
  integer(kind=IGA_INT ) :: i, j, k, l
  integer(kind=IGA_INT ) :: a, b, c, d
  real   (kind=IGA_REAL) :: X1(dim,dim)
  real   (kind=IGA_REAL) :: X2(dim,dim,dim)
  real   (kind=IGA_REAL) :: X3(dim,dim,dim,dim)
  real   (kind=IGA_REAL) :: E1(dim,dim)
  real   (kind=IGA_REAL) :: E2(dim,dim,dim)
  real   (kind=IGA_REAL) :: E3(dim,dim,dim,dim)

  ! 0th derivatives
  R0 = N0
  ! gradient of the mapping
  Jaco = matmul(X,transpose(N1))
  DetJ = Determinant(dim,Jac)
  ! 1st derivatives
  if (ord >= 1) then
     X1 = Jaco
     E1 = Inverse(dim,X1,DetJ)
     R1 = 0
     do idx = 1,nen
        do i = 1,dim
           do a = 1,dim
              R1(i,idx) = N1(a,idx)*E1(a,i) +  R1(i,idx)
           end do
        end do
     end do
  endif
  ! 2nd derivatives
  if (ord >= 2) then
     X2 = 0
     do b = 1,dim
        do a = 1,dim
           do i = 1,dim
              do idx = 1,nen
                 X2(i,a,b) = X(i,idx)*N2(a,b,idx) + X2(i,a,b)
              end do
           end do
        end do
     end do
     E2 = 0
     do j = 1,dim
        do i = 1,dim
           do c = 1,dim
              do b = 1,dim
                 do a = 1,dim
                    do k = 1,dim
                       E2(c,i,j) = - X2(k,a,b)*E1(a,i)*E1(b,j)*E1(c,k) + E2(c,i,j)
                    end do
                 end do
              end do
           end do
        end do
     end do
     R2 = 0
     do idx = 1,nen
        do j = 1,dim
           do i = 1,dim
              do b = 1,dim
                 do a = 1,dim
                    R2(i,j,idx) = N2(a,b,idx)*E1(a,i)*E1(b,j) + R2(i,j,idx)
                 end do
                 R2(i,j,idx) = N1(b,idx)*E2(b,i,j) + R2(i,j,idx)
              end do
           end do
        end do
     end do
  endif
  ! 3rd derivatives
  if (ord >= 3) then
     X3 = 0
     do c = 1,dim
        do b = 1,dim
           do a = 1,dim
              do i = 1,dim
                 do idx = 1,nen
                    X3(i,a,b,c) = X(i,idx)*N3(a,b,c,idx) + X3(i,a,b,c)
                 end do
              end do
           end do
        end do
     end do
     E3 = 0
     do k = 1,dim
        do j = 1,dim
           do i = 1,dim
              do d = 1,dim
                 do a = 1,dim
                    do b = 1,dim
                       do l = 1,dim
                          do c = 1,dim
                             E3(d,i,j,k) = - X3(l,c,b,a)*E1(c,i)*E1(b,j)*E1(a,k)*E1(d,l) + E3(d,i,j,k)
                          end do
                          E3(d,i,j,k) = - X2(l,b,a)*( E1(a,j)*E2(b,i,k) &
                                                    + E1(a,k)*E2(b,i,j) &
                                                    + E1(b,i)*E2(a,j,k) )*E1(d,l) + E3(d,i,j,k)
                       end do
                    end do
                 end do
              end do
           end do
        end do
     end do
     R3 = 0
     do idx = 1,nen
        do k = 1,dim
           do j = 1,dim
              do i = 1,dim
                 do a = 1,dim
                    do b = 1,dim
                       do c = 1,dim
                          R3(i,j,k,idx) = N3(c,b,a,idx)*E1(c,i)*E1(b,j)*E1(a,k) + R3(i,j,k,idx)
                       end do
                       R3(i,j,k,idx) = N2(b,a,idx)*( E1(a,j)*E2(b,i,k) &
                                                   + E1(a,k)*E2(b,i,j) &
                                                   + E1(b,i)*E2(a,j,k) ) + R3(i,j,k,idx)
                    end do
                    R3(i,j,k,idx) = N1(a,idx)*E3(a,i,j,k) + R3(i,j,k,idx)
                 end do
              end do
           end do
        end do
     end do
  end if
  !
end subroutine IsoparametricMapping

pure function Determinant(dim,A) result (detA)
  implicit none
  integer(kind=IGA_INT ), intent(in) :: dim
  real   (kind=IGA_REAL), intent(in) :: A(dim,dim)
  real   (kind=IGA_REAL) :: detA
  select case (dim)
  case (1)
     detA = A(1,1)
  case (2)
     detA = + A(1,1)*A(2,2) - A(2,1)*A(1,2)
  case (3)
     detA = + A(1,1) * ( A(2,2)*A(3,3) - A(3,2)*A(2,3) ) &
            - A(2,1) * ( A(1,2)*A(3,3) - A(3,2)*A(1,3) ) &
            + A(3,1) * ( A(1,2)*A(2,3) - A(2,2)*A(1,3) )
  case default
     detA = 0
  end select
end function Determinant

pure function Inverse(dim,A,detA) result (invA)
  implicit none
  integer(kind=IGA_INT ), intent(in) :: dim
  real   (kind=IGA_REAL), intent(in) :: A(dim,dim)
  real   (kind=IGA_REAL), intent(in) :: detA
  real   (kind=IGA_REAL)             :: invA(dim,dim)
  select case (dim)
  case (1)
     invA = 1.0/detA
  case (2)
     invA(1,1) = + A(2,2)
     invA(2,1) = - A(2,1)
     invA(1,2) = - A(1,2)
     invA(2,2) = + A(1,1)
     invA = invA/detA
  case (3)
     invA(1,1) = + A(2,2)*A(3,3) - A(2,3)*A(3,2)
     invA(2,1) = - A(2,1)*A(3,3) + A(2,3)*A(3,1)
     invA(3,1) = + A(2,1)*A(3,2) - A(2,2)*A(3,1)
     invA(1,2) = - A(1,2)*A(3,3) + A(1,3)*A(3,2)
     invA(2,2) = + A(1,1)*A(3,3) - A(1,3)*A(3,1)
     invA(3,2) = - A(1,1)*A(3,2) + A(1,2)*A(3,1)
     invA(1,3) = + A(1,2)*A(2,3) - A(1,3)*A(2,2)
     invA(2,3) = - A(1,1)*A(2,3) + A(1,3)*A(2,1)
     invA(3,3) = + A(1,1)*A(2,2) - A(1,2)*A(2,1)
     invA = invA/detA
  case default
     invA = 0
  end select
end function Inverse

end subroutine IGA_ShapeFuns
