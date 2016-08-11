pure subroutine IGA_Quadrature_3D(&
     inq,iX,iW,iJ,                &
     jnq,jX,jW,jJ,                &
     knq,kX,kW,kJ,                &
     X,W,J)                       &
  bind(C, name="IGA_Quadrature_3D")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), parameter        :: dim = 3
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: inq
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: jnq
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: knq
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: iX(inq), iW(inq), iJ
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: jX(jnq), jW(jnq), jJ
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: kX(knq), kW(knq), kJ
  real   (kind=IGA_REAL_KIND   ), intent(out) :: X(dim,inq,jnq,knq)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: W(    inq,jnq,knq)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: J(    inq,jnq,knq)
  integer(kind=IGA_INTEGER_KIND) :: iq
  integer(kind=IGA_INTEGER_KIND) :: jq
  integer(kind=IGA_INTEGER_KIND) :: kq
  do kq=1,knq; do jq=1,jnq; do iq=1,inq
     X(1,iq,jq,kq) = iX(iq)
     X(2,iq,jq,kq) = jX(jq)
     X(3,iq,jq,kq) = kX(kq)
     W(  iq,jq,kq) = iW(iq) * jW(jq) * kW(kq)
  end do; end do; end do
  J = iJ * jJ * kJ
end subroutine IGA_Quadrature_3D


pure subroutine IGA_BasisFuns_3D(&
     order,                      &
     inq,ina,iN,                 &
     jnq,jna,jN,                 &
     knq,kna,kN,                 &
     N0,N1,N2,N3,N4)             &
  bind(C, name="IGA_BasisFuns_3D")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), parameter        :: dim = 3
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: order
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: inq, ina
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: jnq, jna
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: knq, kna
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: iN(0:4,ina,inq)
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: jN(0:4,jna,jnq)
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: kN(0:4,kna,knq)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N0(dim**0,ina*jna*kna,inq,jnq,knq)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N1(dim**1,ina*jna*kna,inq,jnq,knq)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N2(dim**2,ina*jna*kna,inq,jnq,knq)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N3(dim**3,ina*jna*kna,inq,jnq,knq)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N4(dim**4,ina*jna*kna,inq,jnq,knq)
  integer(kind=IGA_INTEGER_KIND)  :: iq
  integer(kind=IGA_INTEGER_KIND)  :: jq
  integer(kind=IGA_INTEGER_KIND)  :: kq
  do kq=1,knq; do jq=1,jnq; do iq=1,inq
     call TensorBasisFuns(&
          order,&
          ina,iN(:,:,iq),&
          jna,jN(:,:,jq),&
          kna,kN(:,:,kq),&
          N0(:,:,iq,jq,kq),&
          N1(:,:,iq,jq,kq),&
          N2(:,:,iq,jq,kq),&
          N3(:,:,iq,jq,kq),&
          N4(:,:,iq,jq,kq))
  end do; end do; end do
contains
pure subroutine TensorBasisFuns(&
     order,&
     ina,iN,&
     jna,jN,&
     kna,kN,&
     N0,N1,N2,N3,N4)
  implicit none
  integer(kind=IGA_INTEGER_KIND), parameter        :: dim = 3
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: order
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: ina
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: jna
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: kna
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: iN(0:4,ina)
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: jN(0:4,jna)
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: kN(0:4,kna)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N0(                ina,jna,kna)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N1(            dim,ina,jna,kna)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N2(        dim,dim,ina,jna,kna)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N3(    dim,dim,dim,ina,jna,kna)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N4(dim,dim,dim,dim,ina,jna,kna)
  integer(kind=IGA_INTEGER_KIND)  :: ia, ja, ka
  !
  do ka=1,kna; do ja=1,jna; do ia=1,ina
     N0(ia,ja,ka) = iN(0,ia) * jN(0,ja) * kN(0,ka)
  end do; end do; end do
  !
  if (order < 1) return
  do ka=1,kna; do ja=1,jna; do ia=1,ina
     N1(1,ia,ja,ka) = iN(1,ia) * jN(0,ja) * kN(0,ka)
     N1(2,ia,ja,ka) = iN(0,ia) * jN(1,ja) * kN(0,ka)
     N1(3,ia,ja,ka) = iN(0,ia) * jN(0,ja) * kN(1,ka)
  end do; end do; end do
  !
  if (order < 2) return ! XXX Optimize!
  do ka=1,kna; do ja=1,jna; do ia=1,ina
     N2(1,1,ia,ja,ka) = iN(2,ia) * jN(0,ja) * kN(0,ka)
     N2(2,1,ia,ja,ka) = iN(1,ia) * jN(1,ja) * kN(0,ka)
     N2(3,1,ia,ja,ka) = iN(1,ia) * jN(0,ja) * kN(1,ka)
     N2(1,2,ia,ja,ka) = iN(1,ia) * jN(1,ja) * kN(0,ka)
     N2(2,2,ia,ja,ka) = iN(0,ia) * jN(2,ja) * kN(0,ka)
     N2(3,2,ia,ja,ka) = iN(0,ia) * jN(1,ja) * kN(1,ka)
     N2(1,3,ia,ja,ka) = iN(1,ia) * jN(0,ja) * kN(1,ka)
     N2(2,3,ia,ja,ka) = iN(0,ia) * jN(1,ja) * kN(1,ka)
     N2(3,3,ia,ja,ka) = iN(0,ia) * jN(0,ja) * kN(2,ka)
  end do; end do; end do
  !
  if (order < 3) return ! XXX Optimize!
  do ka=1,kna; do ja=1,jna; do ia=1,ina
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
  end do; end do; end do
  !
  if (order < 4) return ! XXX Optimize!
  do ka=1,kna; do ja=1,jna; do ia=1,ina
     N4(1,1,1,1,ia,ja,ka) = iN(4,ia) * jN(0,ja) * kN(0,ka)
     N4(2,1,1,1,ia,ja,ka) = iN(3,ia) * jN(1,ja) * kN(0,ka)
     N4(3,1,1,1,ia,ja,ka) = iN(3,ia) * jN(0,ja) * kN(1,ka)
     N4(1,2,1,1,ia,ja,ka) = iN(3,ia) * jN(1,ja) * kN(0,ka)
     N4(2,2,1,1,ia,ja,ka) = iN(2,ia) * jN(2,ja) * kN(0,ka)
     N4(3,2,1,1,ia,ja,ka) = iN(2,ia) * jN(1,ja) * kN(1,ka)
     N4(1,3,1,1,ia,ja,ka) = iN(3,ia) * jN(0,ja) * kN(1,ka)
     N4(2,3,1,1,ia,ja,ka) = iN(2,ia) * jN(1,ja) * kN(1,ka)
     N4(3,3,1,1,ia,ja,ka) = iN(2,ia) * jN(0,ja) * kN(2,ka)
     N4(1,1,2,1,ia,ja,ka) = iN(3,ia) * jN(1,ja) * kN(0,ka)
     N4(2,1,2,1,ia,ja,ka) = iN(2,ia) * jN(2,ja) * kN(0,ka)
     N4(3,1,2,1,ia,ja,ka) = iN(2,ia) * jN(1,ja) * kN(1,ka)
     N4(1,2,2,1,ia,ja,ka) = iN(2,ia) * jN(2,ja) * kN(0,ka)
     N4(2,2,2,1,ia,ja,ka) = iN(1,ia) * jN(3,ja) * kN(0,ka)
     N4(3,2,2,1,ia,ja,ka) = iN(1,ia) * jN(2,ja) * kN(1,ka)
     N4(1,3,2,1,ia,ja,ka) = iN(2,ia) * jN(1,ja) * kN(1,ka)
     N4(2,3,2,1,ia,ja,ka) = iN(1,ia) * jN(2,ja) * kN(1,ka)
     N4(3,3,2,1,ia,ja,ka) = iN(1,ia) * jN(1,ja) * kN(2,ka)
     N4(1,1,3,1,ia,ja,ka) = iN(3,ia) * jN(0,ja) * kN(1,ka)
     N4(2,1,3,1,ia,ja,ka) = iN(2,ia) * jN(1,ja) * kN(1,ka)
     N4(3,1,3,1,ia,ja,ka) = iN(2,ia) * jN(0,ja) * kN(2,ka)
     N4(1,2,3,1,ia,ja,ka) = iN(2,ia) * jN(1,ja) * kN(1,ka)
     N4(2,2,3,1,ia,ja,ka) = iN(1,ia) * jN(2,ja) * kN(1,ka)
     N4(3,2,3,1,ia,ja,ka) = iN(1,ia) * jN(1,ja) * kN(2,ka)
     N4(1,3,3,1,ia,ja,ka) = iN(2,ia) * jN(0,ja) * kN(2,ka)
     N4(2,3,3,1,ia,ja,ka) = iN(1,ia) * jN(1,ja) * kN(2,ka)
     N4(3,3,3,1,ia,ja,ka) = iN(1,ia) * jN(0,ja) * kN(3,ka)
     N4(1,1,1,2,ia,ja,ka) = iN(3,ia) * jN(1,ja) * kN(0,ka)
     N4(2,1,1,2,ia,ja,ka) = iN(2,ia) * jN(2,ja) * kN(0,ka)
     N4(3,1,1,2,ia,ja,ka) = iN(2,ia) * jN(1,ja) * kN(1,ka)
     N4(1,2,1,2,ia,ja,ka) = iN(2,ia) * jN(2,ja) * kN(0,ka)
     N4(2,2,1,2,ia,ja,ka) = iN(1,ia) * jN(3,ja) * kN(0,ka)
     N4(3,2,1,2,ia,ja,ka) = iN(1,ia) * jN(2,ja) * kN(1,ka)
     N4(1,3,1,2,ia,ja,ka) = iN(2,ia) * jN(1,ja) * kN(1,ka)
     N4(2,3,1,2,ia,ja,ka) = iN(1,ia) * jN(2,ja) * kN(1,ka)
     N4(3,3,1,2,ia,ja,ka) = iN(1,ia) * jN(1,ja) * kN(2,ka)
     N4(1,1,2,2,ia,ja,ka) = iN(2,ia) * jN(2,ja) * kN(0,ka)
     N4(2,1,2,2,ia,ja,ka) = iN(1,ia) * jN(3,ja) * kN(0,ka)
     N4(3,1,2,2,ia,ja,ka) = iN(1,ia) * jN(2,ja) * kN(1,ka)
     N4(1,2,2,2,ia,ja,ka) = iN(1,ia) * jN(3,ja) * kN(0,ka)
     N4(2,2,2,2,ia,ja,ka) = iN(0,ia) * jN(4,ja) * kN(0,ka)
     N4(3,2,2,2,ia,ja,ka) = iN(0,ia) * jN(3,ja) * kN(1,ka)
     N4(1,3,2,2,ia,ja,ka) = iN(1,ia) * jN(2,ja) * kN(1,ka)
     N4(2,3,2,2,ia,ja,ka) = iN(0,ia) * jN(3,ja) * kN(1,ka)
     N4(3,3,2,2,ia,ja,ka) = iN(0,ia) * jN(2,ja) * kN(2,ka)
     N4(1,1,3,2,ia,ja,ka) = iN(2,ia) * jN(1,ja) * kN(1,ka)
     N4(2,1,3,2,ia,ja,ka) = iN(1,ia) * jN(2,ja) * kN(1,ka)
     N4(3,1,3,2,ia,ja,ka) = iN(1,ia) * jN(1,ja) * kN(2,ka)
     N4(1,2,3,2,ia,ja,ka) = iN(1,ia) * jN(2,ja) * kN(1,ka)
     N4(2,2,3,2,ia,ja,ka) = iN(0,ia) * jN(3,ja) * kN(1,ka)
     N4(3,2,3,2,ia,ja,ka) = iN(0,ia) * jN(2,ja) * kN(2,ka)
     N4(1,3,3,2,ia,ja,ka) = iN(1,ia) * jN(1,ja) * kN(2,ka)
     N4(2,3,3,2,ia,ja,ka) = iN(0,ia) * jN(2,ja) * kN(2,ka)
     N4(3,3,3,2,ia,ja,ka) = iN(0,ia) * jN(1,ja) * kN(3,ka)
     N4(1,1,1,3,ia,ja,ka) = iN(3,ia) * jN(0,ja) * kN(1,ka)
     N4(2,1,1,3,ia,ja,ka) = iN(2,ia) * jN(1,ja) * kN(1,ka)
     N4(3,1,1,3,ia,ja,ka) = iN(2,ia) * jN(0,ja) * kN(2,ka)
     N4(1,2,1,3,ia,ja,ka) = iN(2,ia) * jN(1,ja) * kN(1,ka)
     N4(2,2,1,3,ia,ja,ka) = iN(1,ia) * jN(2,ja) * kN(1,ka)
     N4(3,2,1,3,ia,ja,ka) = iN(1,ia) * jN(1,ja) * kN(2,ka)
     N4(1,3,1,3,ia,ja,ka) = iN(2,ia) * jN(0,ja) * kN(2,ka)
     N4(2,3,1,3,ia,ja,ka) = iN(1,ia) * jN(1,ja) * kN(2,ka)
     N4(3,3,1,3,ia,ja,ka) = iN(1,ia) * jN(0,ja) * kN(3,ka)
     N4(1,1,2,3,ia,ja,ka) = iN(2,ia) * jN(1,ja) * kN(1,ka)
     N4(2,1,2,3,ia,ja,ka) = iN(1,ia) * jN(2,ja) * kN(1,ka)
     N4(3,1,2,3,ia,ja,ka) = iN(1,ia) * jN(1,ja) * kN(2,ka)
     N4(1,2,2,3,ia,ja,ka) = iN(1,ia) * jN(2,ja) * kN(1,ka)
     N4(2,2,2,3,ia,ja,ka) = iN(0,ia) * jN(3,ja) * kN(1,ka)
     N4(3,2,2,3,ia,ja,ka) = iN(0,ia) * jN(2,ja) * kN(2,ka)
     N4(1,3,2,3,ia,ja,ka) = iN(1,ia) * jN(1,ja) * kN(2,ka)
     N4(2,3,2,3,ia,ja,ka) = iN(0,ia) * jN(2,ja) * kN(2,ka)
     N4(3,3,2,3,ia,ja,ka) = iN(0,ia) * jN(1,ja) * kN(3,ka)
     N4(1,1,3,3,ia,ja,ka) = iN(2,ia) * jN(0,ja) * kN(2,ka)
     N4(2,1,3,3,ia,ja,ka) = iN(1,ia) * jN(1,ja) * kN(2,ka)
     N4(3,1,3,3,ia,ja,ka) = iN(1,ia) * jN(0,ja) * kN(3,ka)
     N4(1,2,3,3,ia,ja,ka) = iN(1,ia) * jN(1,ja) * kN(2,ka)
     N4(2,2,3,3,ia,ja,ka) = iN(0,ia) * jN(2,ja) * kN(2,ka)
     N4(3,2,3,3,ia,ja,ka) = iN(0,ia) * jN(1,ja) * kN(3,ka)
     N4(1,3,3,3,ia,ja,ka) = iN(1,ia) * jN(0,ja) * kN(3,ka)
     N4(2,3,3,3,ia,ja,ka) = iN(0,ia) * jN(1,ja) * kN(3,ka)
     N4(3,3,3,3,ia,ja,ka) = iN(0,ia) * jN(0,ja) * kN(4,ka)
  end do; end do; end do
  !
end subroutine TensorBasisFuns
end subroutine IGA_BasisFuns_3D


pure subroutine IGA_Rationalize_3D(&
     order,                      &
     nqp,nen,W,                  &
     N0,N1,N2,N3,N4)             &
  bind(C, name="IGA_Rationalize_3D")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), parameter        :: dim = 3
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: order
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: nen
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: nqp
  real   (kind=IGA_REAL_KIND   ), intent(in)    :: W(nen)
  real   (kind=IGA_REAL_KIND   ), intent(inout) :: N0(dim**0,nen,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(inout) :: N1(dim**1,nen,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(inout) :: N2(dim**2,nen,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(inout) :: N3(dim**3,nen,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(inout) :: N4(dim**4,nen,nqp)
  integer(kind=IGA_INTEGER_KIND)  :: q
  do q=1,nqp
     call Rationalize(&
          order,&
          nen,W,&
          N0(:,:,q),&
          N1(:,:,q),&
          N2(:,:,q),&
          N3(:,:,q),&
          N4(:,:,q))
  end do
contains
include 'petigarat.f90.in'
end subroutine IGA_Rationalize_3D


pure subroutine IGA_GeometryMap_3D(&
     order,                        &
     nqp,nen,X,                    &
     M0,M1,M2,M3,M4,               &
     X0,X1,X2,X3,X4)               &
  bind(C, name="IGA_GeometryMap_3D")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), parameter        :: dim = 3
  integer(kind=IGA_INTEGER_KIND), parameter        :: nsd = 3
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: order
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
end subroutine IGA_GeometryMap_3D


pure subroutine IGA_InverseMap_3D(&
     order,                       &
     nqp,                         &
     X1,X2,X3,X4,                 &
     dX,                          &
     E1,E2,E3,E4)                 &
  bind(C, name="IGA_InverseMap_3D")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), parameter        :: dim = 3
  integer(kind=IGA_INTEGER_KIND), parameter        :: nsd = 3
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: order
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: nqp
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: X1(dim**1*nsd,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: X2(dim**2*nsd,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: X3(dim**3*nsd,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: X4(dim**4*nsd,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: dX(nqp)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: E1(nsd**1*dim,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: E2(nsd**2*dim,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: E3(nsd**3*dim,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: E4(nsd**4*dim,nqp)
  integer(kind=IGA_INTEGER_KIND)  :: q
  do q=1,nqp
     call InverseMap(&
          order,&
          X1(:,q),X2(:,q),X3(:,q),X4(:,q),&
          dX(q),&
          E1(:,q),E2(:,q),E3(:,q),E4(:,q))
  end do
contains
include 'petigamapinv.f90.in'
end subroutine IGA_InverseMap_3D


pure subroutine IGA_ShapeFuns_3D(&
     order,                      &
     nqp,nen,                    &
     E1,E2,E3,E4,                &
     M1,M2,M3,M4,                &
     N1,N2,N3,N4)                &
  bind(C, name="IGA_ShapeFuns_3D")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), parameter        :: dim = 3
  integer(kind=IGA_INTEGER_KIND), parameter        :: nsd = 3
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: order
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: nqp
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: nen
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: E1(nsd**1*dim,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: E2(nsd**2*dim,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: E3(nsd**3*dim,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: E4(nsd**4*dim,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: M1(dim**1*nen,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: M2(dim**2*nen,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: M3(dim**3*nen,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: M4(dim**4*nen,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N1(nsd**1*nen,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N2(nsd**2*nen,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N3(nsd**3*nen,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N4(nsd**4*nen,nqp)
  integer(kind=IGA_INTEGER_KIND)  :: q
  do q=1,nqp
     call ShapeFunctions(&
          order,nen,&
          E1(:,q),E2(:,q),E3(:,q),E4(:,q),&
          M1(:,q),M2(:,q),M3(:,q),M4(:,q),&
          N1(:,q),N2(:,q),N3(:,q),N4(:,q))
  end do
contains
include 'petigamapshf.f90.in'
end subroutine IGA_ShapeFuns_3D


pure subroutine IGA_BoundaryArea_3D(&
     m,axis,side,              &
     geometry,Cx,              &
     rational,Cw,              &
     inqp,iW,inen,iN,          &
     jnqp,jW,jnen,jN,          &
     dS)                       &
  bind(C, name="IGA_BoundaryArea_3D")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), parameter         :: nsd = 3
  integer(kind=IGA_INTEGER_KIND), parameter         :: dim = 2
  integer(kind=IGA_INTEGER_KIND), intent(in)        :: m(3)
  integer(kind=IGA_INTEGER_KIND), intent(in),value  :: axis, side
  integer(kind=IGA_INTEGER_KIND), intent(in),value  :: geometry, rational
  real   (kind=IGA_REAL_KIND   ), intent(in)        :: Cx(nsd,m(1),m(2),m(3))
  real   (kind=IGA_REAL_KIND   ), intent(in)        :: Cw(    m(1),m(2),m(3))
  integer(kind=IGA_INTEGER_KIND), intent(in),value  :: inqp, inen
  integer(kind=IGA_INTEGER_KIND), intent(in),value  :: jnqp, jnen
  real   (kind=IGA_REAL_KIND   ), intent(in)        :: iW(inqp), iN(0:4,inen,inqp)
  real   (kind=IGA_REAL_KIND   ), intent(in)        :: jW(jnqp), jN(0:4,jnen,jnqp)
  real   (kind=IGA_REAL_KIND   ), intent(out)       :: dS
  integer(kind=IGA_INTEGER_KIND)  :: k, nen, iq, jq, ia, ja
  real   (kind=IGA_REAL_KIND   )  :: N0(inen,jnen), N1(dim,inen,jnen), detJ
  real   (kind=IGA_REAL_KIND   )  :: Xw(inen,jnen), Xx(nsd,inen,jnen)
  nen = inen*jnen
  select case (axis)
  case (0)
     if (side==0) k=1
     if (side==1) k=m(1)
     Xx = Cx(:,k,:,:); Xw = Cw(k,:,:)
  case (1)
     if (side==0) k=1
     if (side==1) k=m(2)
     Xx = Cx(:,:,k,:); Xw = Cw(:,k,:)
  case (2)
     if (side==0) k=1
     if (side==1) k=m(3)
     Xx = Cx(:,:,:,k); Xw = Cw(:,:,k)
  end select
  detJ = 1
  dS = 0
  do jq=1,jnqp
     do iq=1,inqp
        do ja=1,jnen; do ia=1,inen
           N0(ia,ja) = iN(0,ia,iq) * jN(0,ja,jq)
        end do; end do
        do ja=1,jnen; do ia=1,inen
           N1(1,ia,ja) = iN(1,ia,iq) * jN(0,ja,jq)
           N1(2,ia,ja) = iN(0,ia,iq) * jN(1,ja,jq)
        end do; end do
        if (rational /= 0) call Rationalize(nen,Xw,N0,N1)
        if (geometry /= 0) call Jacobian(nen,N1,Xx,detJ)
        dS = dS + detJ * iW(iq)*jW(jq)
     end do
  end do
contains
pure subroutine Rationalize(nen,W,R0,R1)
  implicit none
  integer(kind=IGA_INTEGER_KIND), intent(in)    :: nen
  real   (kind=IGA_REAL_KIND   ), intent(in)    :: W(nen)
  real   (kind=IGA_REAL_KIND   ), intent(inout) :: R0(    nen)
  real   (kind=IGA_REAL_KIND   ), intent(inout) :: R1(dim,nen)
  integer(kind=IGA_INTEGER_KIND)  :: i
  real   (kind=IGA_REAL_KIND   )  :: W0
  R0 = W * R0
  W0 = sum(R0)
  R0 = R0 / W0
  do i=1,dim
  R1(i,:) = W*R1(i,:) - R0 * sum(W*R1(i,:))
  end do
  R1 = R1 / W0
end subroutine Rationalize
pure subroutine Jacobian(nen,N,X,J)
  implicit none
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: nen
  real   (kind=IGA_REAL_KIND   ), intent(in)       :: N(dim,nen)
  real   (kind=IGA_REAL_KIND   ), intent(in)       :: X(nsd,nen)
  real   (kind=IGA_REAL_KIND   ), intent(out)      :: J
  real   (kind=IGA_REAL_KIND   )  :: F(dim,nsd), M(dim,dim)
  F = matmul(N,transpose(X))
  M = matmul(F,transpose(F))
  J = sqrt(abs(Determinant(dim,M)))
end subroutine Jacobian
include 'petigadet.f90.in'
end subroutine IGA_BoundaryArea_3D
