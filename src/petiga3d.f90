subroutine IGA_Quadrature_3D(&
     inq,iX,iW, &
     jnq,jX,jW, &
     knq,kX,kW, &
     X, W)      &
  bind(C, name="IGA_Quadrature_3D")
  use PetIGA
  implicit none
  integer(kind=IGA_INT ), parameter        :: dim = 3
  integer(kind=IGA_INT ), intent(in),value :: inq
  integer(kind=IGA_INT ), intent(in),value :: jnq
  integer(kind=IGA_INT ), intent(in),value :: knq
  real   (kind=IGA_REAL), intent(in)  :: iX(inq), iW(inq)
  real   (kind=IGA_REAL), intent(in)  :: jX(jnq), jW(jnq)
  real   (kind=IGA_REAL), intent(in)  :: kX(knq), kW(knq)
  real   (kind=IGA_REAL), intent(out) :: X(dim,inq,jnq,knq)
  real   (kind=IGA_REAL), intent(out) :: W(inq,jnq,knq)
  integer(kind=IGA_INT ) :: iq
  integer(kind=IGA_INT ) :: jq
  integer(kind=IGA_INT ) :: kq
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
  use PetIGA
  implicit none
  integer(kind=IGA_INT ), parameter        :: dim = 3
  integer(kind=IGA_INT ), intent(in),value :: geometry
  integer(kind=IGA_INT ), intent(in),value :: rational
  integer(kind=IGA_INT ), intent(in),value :: inq, ina, ind
  integer(kind=IGA_INT ), intent(in),value :: jnq, jna, jnd
  integer(kind=IGA_INT ), intent(in),value :: knq, kna, knd
  real   (kind=IGA_REAL), intent(in)  :: iJ, iN(0:ind,ina,inq)
  real   (kind=IGA_REAL), intent(in)  :: jJ, jN(0:jnd,jna,jnq)
  real   (kind=IGA_REAL), intent(in)  :: kJ, kN(0:knd,kna,knq)
  real   (kind=IGA_REAL), intent(in)  :: Cw(dim+1,ina,jna,kna)
  real   (kind=IGA_REAL), intent(out) :: detJac(     inq,jnq,knq)
  real   (kind=IGA_REAL), intent(out) :: Jac(dim,dim,inq,jnq,knq)
  real   (kind=IGA_REAL), intent(out) :: N0(       ina,jna,kna,inq,jnq,knq)
  real   (kind=IGA_REAL), intent(out) :: N1(   dim,ina,jna,kna,inq,jnq,knq)
  real   (kind=IGA_REAL), intent(out) :: N2(dim**2,ina,jna,kna,inq,jnq,knq)
  real   (kind=IGA_REAL), intent(out) :: N3(dim**3,ina,jna,kna,inq,jnq,knq)

  integer(kind=IGA_INT ) :: ia,iq
  integer(kind=IGA_INT ) :: ja,jq
  integer(kind=IGA_INT ) :: ka,kq
  integer(kind=IGA_INT ) :: i,na,nd
  real   (kind=IGA_REAL) :: C(dim,ina,jna,kna)
  real   (kind=IGA_REAL) :: w(    ina,jna,kna)

  if (geometry /= 0) then
     w = Cw(dim+1,:,:,:)
     forall (i=1:dim)
        C(i,:,:,:) = Cw(i,:,:,:) / w
     end forall
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

pure subroutine TensorBasisFuns(&
     ina,ind,iN,&
     jna,jnd,jN,&
     kna,knd,kN,&
     nd,N0,N1,N2,N3)
  implicit none
  integer(kind=IGA_INT ), parameter        :: dim = 3
  integer(kind=IGA_INT ), intent(in),value :: ina, ind
  integer(kind=IGA_INT ), intent(in),value :: jna, jnd
  integer(kind=IGA_INT ), intent(in),value :: kna, knd
  real   (kind=IGA_REAL), intent(in)  :: iN(0:ind,ina)
  real   (kind=IGA_REAL), intent(in)  :: jN(0:jnd,jna)
  real   (kind=IGA_REAL), intent(in)  :: kN(0:knd,kna)
  integer(kind=IGA_INT ), intent(in)  :: nd
  real   (kind=IGA_REAL), intent(out) :: N0(            ina,jna,kna)
  real   (kind=IGA_REAL), intent(out) :: N1(        dim,ina,jna,kna)
  real   (kind=IGA_REAL), intent(out) :: N2(    dim,dim,ina,jna,kna)
  real   (kind=IGA_REAL), intent(out) :: N3(dim,dim,dim,ina,jna,kna)
  integer(kind=IGA_INT ) :: ia, ja, ka
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
include 'petigageo.f90.in'

end subroutine IGA_ShapeFuns_3D
