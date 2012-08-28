pure subroutine IGA_Quadrature_3D(&
     inq,iX,iW,iL,           &
     jnq,jX,jW,jL,           &
     knq,kX,kW,kL,           &
     W,J,X,L)                &
  bind(C, name="IGA_Quadrature_3D")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), parameter        :: dim = 3
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: inq
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: jnq
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: knq
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: iX(inq), iW(inq), iL
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: jX(jnq), jW(jnq), jL
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: kX(knq), kW(knq), kL
  real   (kind=IGA_REAL_KIND   ), intent(out) :: W(    inq,jnq,knq)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: J(    inq,jnq,knq)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: X(dim,inq,jnq,knq)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: L(dim,inq,jnq,knq)
  integer(kind=IGA_INTEGER_KIND) :: iq
  integer(kind=IGA_INTEGER_KIND) :: jq
  integer(kind=IGA_INTEGER_KIND) :: kq
  forall (iq=1:inq, jq=1:jnq, kq=1:knq)
     !
     W(iq,jq,kq) = iW(iq) * jW(jq) * kW(kq)
     J(iq,jq,kq) = iL * jL * kL
     !
     X(1,iq,jq,kq) = iX(iq)
     X(2,iq,jq,kq) = jX(jq)
     X(3,iq,jq,kq) = kX(kq)
     L(1,iq,jq,kq) = iL
     L(2,iq,jq,kq) = jL
     L(3,iq,jq,kq) = kL
  end forall
end subroutine IGA_Quadrature_3D


pure subroutine IGA_BasisFuns_3D(&
     order,                 &
     rational,W,            &
     inq,ina,ind,iN,        &
     jnq,jna,jnd,jN,        &
     knq,kna,knd,kN,        &
     N0,N1,N2,N3)           &
  bind(C, name="IGA_BasisFuns_3D")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), parameter        :: dim = 3
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: order
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: rational
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: inq, ina, ind
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: jnq, jna, jnd
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: knq, kna, knd
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: iN(0:ind,ina,inq)
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: jN(0:jnd,jna,jnq)
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: kN(0:knd,kna,knq)
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: W(dim+1,  ina*jna*kna)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N0(       ina*jna*kna,inq,jnq,knq)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N1(   dim,ina*jna*kna,inq,jnq,knq)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N2(dim**2,ina*jna*kna,inq,jnq,knq)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N3(dim**3,ina*jna*kna,inq,jnq,knq)
  integer(kind=IGA_INTEGER_KIND)  :: ia, iq
  integer(kind=IGA_INTEGER_KIND)  :: ja, jq
  integer(kind=IGA_INTEGER_KIND)  :: ka, kq
  integer(kind=IGA_INTEGER_KIND)  :: nen
  nen = ina*jna*kna
  do kq=1,knq
     do jq=1,jnq
        do iq=1,inq
           call TensorBasisFuns(&
                order,&
                ina,ind,iN(:,:,iq),&
                jna,jnd,jN(:,:,jq),&
                kna,knd,kN(:,:,kq),&
                N0(  :,iq,jq,kq),&
                N1(:,:,iq,jq,kq),&
                N2(:,:,iq,jq,kq),&
                N3(:,:,iq,jq,kq))
           if (rational /= 0) then
              call Rationalize(&
                   order,&
                   nen,W,&
                   N0(  :,iq,jq,kq),&
                   N1(:,:,iq,jq,kq),&
                   N2(:,:,iq,jq,kq),&
                   N3(:,:,iq,jq,kq))
           end if
        end do
     end do
  end do

contains

pure subroutine TensorBasisFuns(&
     ord,&
     ina,ind,iN,&
     jna,jnd,jN,&
     kna,knd,kN,&
     N0,N1,N2,N3)
  implicit none
  integer(kind=IGA_INTEGER_KIND), parameter        :: dim = 3
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: ord
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: ina, ind
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: jna, jnd
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: kna, knd
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: iN(0:ind,ina)
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: jN(0:jnd,jna)
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: kN(0:knd,kna)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N0(            ina,jna,kna)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N1(        dim,ina,jna,kna)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N2(    dim,dim,ina,jna,kna)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N3(dim,dim,dim,ina,jna,kna)
  integer(kind=IGA_INTEGER_KIND)  :: ia, ja, ka
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
  if (ord < 2) return ! XXX Optimize!
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
  if (ord < 3) return ! XXX Optimize!
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

end subroutine IGA_BasisFuns_3D


pure subroutine IGA_ShapeFuns_3D(&
     order,                 &
     nqp,nen,X,             &
     M0,M1,M2,M3,           &
     N0,N1,N2,N3,           &
     DetF,F,G)              &
  bind(C, name="IGA_ShapeFuns_3D")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), parameter        :: dim = 3
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: order
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: nqp
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: nen
  real   (kind=IGA_REAL_KIND   ), intent(in)    :: X(dim+1,nen)
  real   (kind=IGA_REAL_KIND   ), intent(in)    :: M0(       nen,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(in)    :: M1(dim,   nen,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(in)    :: M2(dim**2,nen,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(in)    :: M3(dim**3,nen,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(out)   :: N0(       nen,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(out)   :: N1(dim,   nen,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(out)   :: N2(dim**2,nen,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(out)   :: N3(dim**3,nen,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(out)   :: DetF(nqp)
  real   (kind=IGA_REAL_KIND   ), intent(out)   :: F(dim,dim,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(out)   :: G(dim,dim,nqp)
  integer(kind=IGA_INTEGER_KIND)  :: q
  do q=1,nqp
     call GeometryMap(&
          order,&
          nen,X,&
          M0(:,q),M1(:,:,q),M2(:,:,q),M3(:,:,q),&
          N0(:,q),N1(:,:,q),N2(:,:,q),N3(:,:,q),&
          DetF(q),F(:,:,q),G(:,:,q))
  end do
contains
include 'petigageo.f90.in'
end subroutine IGA_ShapeFuns_3D
