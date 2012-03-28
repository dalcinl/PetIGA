subroutine IGA_Quadrature_3D(&
     inq,iX,iW, &
     jnq,jX,jW, &
     knq,kX,kW, &
     X, W)      &
  bind(C, name="IGA_Quadrature_3D")
  use ISO_C_BINDING, only: C_INT, C_LONG
  use ISO_C_BINDING, only: C_FLOAT, C_DOUBLE
  implicit none
  integer(kind=C_INT   ), parameter        :: dim = 3
  integer(kind=C_INT   ), intent(in),value :: inq
  integer(kind=C_INT   ), intent(in),value :: jnq
  integer(kind=C_INT   ), intent(in),value :: knq
  real   (kind=C_DOUBLE), intent(in)  :: iX(inq), iW(inq)
  real   (kind=C_DOUBLE), intent(in)  :: jX(jnq), jW(jnq)
  real   (kind=C_DOUBLE), intent(in)  :: kX(knq), kW(knq)
  real   (kind=C_DOUBLE), intent(out) :: X(dim,inq,jnq,knq)
  real   (kind=C_DOUBLE), intent(out) :: W(inq,jnq,knq)
  integer :: iq
  integer :: jq
  integer :: kq
  forall (iq=1:inq, jq=1:jnq, kq=1:knq)
     X(:,iq,jq,kq) = (/ iX(iq),  jX(jq),  kX(kq) /)
     W(  iq,jq,kq) =    iW(iq) * jW(jq) * kW(kq)
  end forall
end subroutine IGA_Quadrature_3D

subroutine IGA_ShapeFuns_3D(&
     geometry,rational,     &
     inq,ina,ind,iJ,iN,     &
     jnq,jna,jnd,jJ,jN,     &
     knq,kna,knd,kJ,kN,     &
     Cw,detJac,Jac,         &
     N0,N1,N2,N3)           &
  bind(C, name="IGA_ShapeFuns_3D")
  use ISO_C_BINDING, only: C_INT, C_LONG
  use ISO_C_BINDING, only: C_FLOAT, C_DOUBLE
  implicit none
  integer(kind=C_INT   ), parameter        :: dim = 3
  integer(kind=C_INT   ), intent(in),value :: geometry
  integer(kind=C_INT   ), intent(in),value :: rational
  integer(kind=C_INT   ), intent(in),value :: inq, ina, ind
  integer(kind=C_INT   ), intent(in),value :: jnq, jna, jnd
  integer(kind=C_INT   ), intent(in),value :: knq, kna, knd
  real   (kind=C_DOUBLE), intent(in)  :: iJ, iN(0:ind,ina,inq)
  real   (kind=C_DOUBLE), intent(in)  :: jJ, jN(0:jnd,jna,jnq)
  real   (kind=C_DOUBLE), intent(in)  :: kJ, kN(0:knd,kna,knq)
  real   (kind=C_DOUBLE), intent(in)  :: Cw(dim+1,ina,jna,kna)
  real   (kind=C_DOUBLE), intent(out) :: detJac(     inq,jnq,knq)
  real   (kind=C_DOUBLE), intent(out) :: Jac(dim,dim,inq,jnq,knq)
  real   (kind=C_DOUBLE), intent(out) :: N0(       ina,jna,kna,inq,jnq,knq)
  real   (kind=C_DOUBLE), intent(out) :: N1(   dim,ina,jna,kna,inq,jnq,knq)
  real   (kind=C_DOUBLE), intent(out) :: N2(dim**2,ina,jna,kna,inq,jnq,knq)
  real   (kind=C_DOUBLE), intent(out) :: N3(dim**3,ina,jna,kna,inq,jnq,knq)

  integer :: ia,iq
  integer :: ja,jq
  integer :: ka,kq
  integer :: na,nd

  real(kind=C_DOUBLE) :: C(dim,ina,jna,kna)
  real(kind=C_DOUBLE) :: w(    ina,jna,kna)
  if (geometry /= 0) then
     C = Cw(1:dim,:,:,:)
  end if
  if (rational /= 0) then
     w = Cw(dim+1,:,:,:) 
  end if

  nd = max(1,min(ind,jnd,knd,3))
  na = ina*jna*kna
  do kq=1,knq
     do jq=1,jnq
        do iq=1,inq
           call TensorBasisFuns(&
                ina,ind,iN(:,:,iq),&
                jna,jnd,jN(:,:,jq),&
                kna,knd,kN(:,:,kq),&
                nd,&
                N0(  :,:,:,iq,jq,kq),&
                N1(:,:,:,:,iq,jq,kq),&
                N2(:,:,:,:,iq,jq,kq),&
                N3(:,:,:,:,iq,jq,kq))
           if (rational /= 0) then
              call Rationalize(&
                   nd,na,w,&
                   N0(  :,:,:,iq,jq,kq),&
                   N1(:,:,:,:,iq,jq,kq),&
                   N2(:,:,:,:,iq,jq,kq),&
                   N3(:,:,:,:,iq,jq,kq))
           end if
           if (geometry /= 0) then
              call GeometryMap(&
                   nd,na,C,&
                   detJac( iq,jq,kq),&
                   Jac(:,:,iq,jq,kq),&
                   N0(  :,:,:,iq,jq,kq),&
                   N1(:,:,:,:,iq,jq,kq),&
                   N2(:,:,:,:,iq,jq,kq),&
                   N3(:,:,:,:,iq,jq,kq))
              detJac( iq,jq,kq) = detJac( iq,jq,kq) * (iJ*jJ*kJ)
              Jac(1,:,iq,jq,kq) = Jac(1,:,iq,jq,kq) * iJ
              Jac(2,:,iq,jq,kq) = Jac(2,:,iq,jq,kq) * jJ
              Jac(3,:,iq,jq,kq) = Jac(3,:,iq,jq,kq) * kJ
           else
              detJac( iq,jq,kq) = (iJ*jJ*kJ)
              Jac(:,:,iq,jq,kq) = 0
              Jac(1,1,iq,jq,kq) = iJ
              Jac(2,2,iq,jq,kq) = jJ
              Jac(3,3,iq,jq,kq) = kJ
           end if
        end do
     end do
  end do

contains

subroutine TensorBasisFuns(&
     ina,ind,iN,&
     jna,jnd,jN,&
     kna,knd,kN,&
     nd,N0,N1,N2,N3)
  use ISO_C_BINDING, only: C_INT, C_LONG
  use ISO_C_BINDING, only: C_FLOAT, C_DOUBLE
  implicit none
  integer(kind=C_INT   ), parameter        :: dim = 3
  integer(kind=C_INT   ), intent(in),value :: ina, ind
  integer(kind=C_INT   ), intent(in),value :: jna, jnd
  integer(kind=C_INT   ), intent(in),value :: kna, knd
  real   (kind=C_DOUBLE), intent(in)  :: iN(0:ind,ina)
  real   (kind=C_DOUBLE), intent(in)  :: jN(0:jnd,jna)
  real   (kind=C_DOUBLE), intent(in)  :: kN(0:knd,kna)
  integer(kind=C_INT   ), intent(in)  :: nd
  real   (kind=C_DOUBLE), intent(out) :: N0(            ina,jna,kna)
  real   (kind=C_DOUBLE), intent(out) :: N1(        dim,ina,jna,kna)
  real   (kind=C_DOUBLE), intent(out) :: N2(    dim,dim,ina,jna,kna)
  real   (kind=C_DOUBLE), intent(out) :: N3(dim,dim,dim,ina,jna,kna)
  integer :: ia, ja, ka
  !
  forall (ia=1:ina, ja=1:jna, ka=1:kna)
     N0(ia,ja,ka) = iN(0,ia) * jN(0,ja) * kN(0,ka)
  end forall
  !
  forall (ia=1:ina, ja=1:jna, ka=1:kna)
     N1(1,ia,ja,ka) = iN(1,ia) * jN(0,ja) * kN(0,ka)
     N1(2,ia,ja,ka) = iN(0,ia) * jN(1,ja) * kN(0,ka)
     N1(3,ia,ja,ka) = iN(0,ia) * jN(0,ja) * kN(1,ka)
  end forall
  !
  if (nd < 2) return ! XXX Optimize!
  forall (ia=1:ina, ja=1:jna, ka=1:kna)
     N2(1,1,ia,ja,ka) = iN(2,ia) * jN(0,ja) * kN(0,ka)
     N2(2,1,ia,ja,ka) = iN(1,ia) * jN(1,ja) * kN(0,ka)
     N2(3,1,ia,ja,ka) = iN(1,ia) * jN(0,ja) * kN(1,ka)
     N2(1,2,ia,ja,ka) = iN(1,ia) * jN(1,ja) * kN(0,ka)
     N2(2,2,ia,ja,ka) = iN(0,ia) * jN(2,ja) * kN(0,ka)
     N2(3,2,ia,ja,ka) = iN(0,ia) * jN(1,ja) * kN(1,ka)
     N2(1,3,ia,ja,ka) = iN(1,ia) * jN(0,ja) * kN(1,ka)
     N2(2,3,ia,ja,ka) = iN(0,ia) * jN(1,ja) * kN(1,ka)
     N2(3,3,ia,ja,ka) = iN(0,ia) * jN(0,ja) * kN(2,ka)
  end forall
  !
  if (nd < 3) return ! XXX Optimize!
  forall (ia=1:ina, ja=1:jna, ka=1:kna)
     N3(1,1,1,ia,ja,ka) = iN(3,ia) * jN(0,ja) * kN(0,ka)
     N3(2,1,1,ia,ja,ka) = iN(2,ia) * jN(1,ja) * kN(0,ka)
     N3(3,1,1,ia,ja,ka) = iN(2,ia) * jN(0,ja) * kN(1,ka)
     N3(1,2,1,ia,ja,ka) = iN(2,ia) * jN(1,ja) * kN(0,ka)
     N3(2,2,1,ia,ja,ka) = iN(1,ia) * jN(2,ja) * kN(0,ka)
     N3(3,2,1,ia,ja,ka) = iN(1,ia) * jN(1,ja) * kN(1,ka)
     N3(1,3,1,ia,ja,ka) = iN(2,ia) * jN(0,ja) * kN(1,ka)
     N3(2,3,1,ia,ja,ka) = iN(1,ia) * jN(1,ja) * kN(1,ka)
     N3(3,3,1,ia,ja,ka) = iN(1,ia) * jN(0,ja) * kN(2,ka)
     N3(1,1,2,ia,ja,ka) = iN(2,ia) * jN(1,ja) * kN(0,ka)
     N3(2,1,2,ia,ja,ka) = iN(1,ia) * jN(2,ja) * kN(0,ka)
     N3(3,1,2,ia,ja,ka) = iN(1,ia) * jN(1,ja) * kN(1,ka)
     N3(1,2,2,ia,ja,ka) = iN(1,ia) * jN(2,ja) * kN(0,ka)
     N3(2,2,2,ia,ja,ka) = iN(0,ia) * jN(3,ja) * kN(0,ka)
     N3(3,2,2,ia,ja,ka) = iN(0,ia) * jN(2,ja) * kN(1,ka)
     N3(1,3,2,ia,ja,ka) = iN(1,ia) * jN(1,ja) * kN(1,ka)
     N3(2,3,2,ia,ja,ka) = iN(0,ia) * jN(2,ja) * kN(1,ka)
     N3(3,3,2,ia,ja,ka) = iN(0,ia) * jN(1,ja) * kN(2,ka)
     N3(1,1,3,ia,ja,ka) = iN(2,ia) * jN(0,ja) * kN(1,ka)
     N3(2,1,3,ia,ja,ka) = iN(1,ia) * jN(1,ja) * kN(1,ka)
     N3(3,1,3,ia,ja,ka) = iN(1,ia) * jN(0,ja) * kN(2,ka)
     N3(1,2,3,ia,ja,ka) = iN(1,ia) * jN(1,ja) * kN(1,ka)
     N3(2,2,3,ia,ja,ka) = iN(0,ia) * jN(2,ja) * kN(1,ka)
     N3(3,2,3,ia,ja,ka) = iN(0,ia) * jN(1,ja) * kN(2,ka)
     N3(1,3,3,ia,ja,ka) = iN(1,ia) * jN(0,ja) * kN(2,ka)
     N3(2,3,3,ia,ja,ka) = iN(0,ia) * jN(1,ja) * kN(2,ka)
     N3(3,3,3,ia,ja,ka) = iN(0,ia) * jN(0,ja) * kN(3,ka)
  end forall
  !
end subroutine TensorBasisFuns

include 'petigarat.f90.in'
include 'petigamap.f90.in'

end subroutine IGA_ShapeFuns_3D
