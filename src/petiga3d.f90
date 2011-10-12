subroutine IGA_Quadrature_3D(&
     inq,iX,iW, &
     jnq,jX,jW, &
     knq,kX,kW, &
     X, W)      &
  bind(C, name="IGA_Quadrature_3D")
  use ISO_C_BINDING, only: C_INT, C_LONG
  use ISO_C_BINDING, only: C_FLOAT, C_DOUBLE
  implicit none
  integer, parameter :: dim = 3
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
     inq,ina,ind,iX,iJ,iN, &
     jnq,jna,jnd,jX,jJ,jN, &
     knq,kna,knd,kX,kJ,kN, &
     detJ,X,J,N0,N1,N2,N3) &
  bind(C, name="IGA_ShapeFuns_3D")
  use ISO_C_BINDING, only: C_INT, C_LONG
  use ISO_C_BINDING, only: C_FLOAT, C_DOUBLE
  implicit none
  integer(kind=C_INT   ), parameter        :: dim = 3
  integer(kind=C_INT   ), intent(in),value :: inq, ina, ind
  integer(kind=C_INT   ), intent(in),value :: jnq, jna, jnd
  integer(kind=C_INT   ), intent(in),value :: knq, kna, knd
  real   (kind=C_DOUBLE), intent(in)  :: iX(inq), iJ, iN(0:ind,ina,inq)
  real   (kind=C_DOUBLE), intent(in)  :: jX(jnq), jJ, jN(0:jnd,jna,jnq)
  real   (kind=C_DOUBLE), intent(in)  :: kX(knq), kJ, kN(0:knd,kna,knq)
  real   (kind=C_DOUBLE), intent(out) :: detJ(     inq,jnq,knq)
  real   (kind=C_DOUBLE), intent(out) :: X(    dim,inq,jnq,knq)
  real   (kind=C_DOUBLE), intent(out) :: J(dim,dim,inq,jnq,knq)
  real   (kind=C_DOUBLE), intent(out) :: N0(       ina,jna,kna,inq,jnq,knq)
  real   (kind=C_DOUBLE), intent(out) :: N1(   dim,ina,jna,kna,inq,jnq,knq)
  real   (kind=C_DOUBLE), intent(out) :: N2(dim**2,ina,jna,kna,inq,jnq,knq)
  real   (kind=C_DOUBLE), intent(out) :: N3(dim**3,ina,jna,kna,inq,jnq,knq)
  integer :: i
  integer, parameter :: d2(dim,dim) = reshape((/(i,i=1,dim*dim)/),(/dim,dim/))
  integer :: ia,iq
  integer :: ja,jq
  integer :: ka,kq

  forall (iq=1:inq, jq=1:jnq, kq=1:knq)
     forall (ia=1:ina, ja=1:jna, ka=1:kna)
        !
        N0(ia,ja,ka,iq,jq,kq)   = iN(0,ia,iq) * jN(0,ja,jq) * kN(0,ka,kq)
        !
        N1(1,ia,ja,ka,iq,jq,kq) = iN(1,ia,iq) * jN(0,ja,jq) * kN(0,ka,kq)
        N1(2,ia,ja,ka,iq,jq,kq) = iN(0,ia,iq) * jN(1,ja,jq) * kN(0,ka,kq)
        N1(3,ia,ja,ka,iq,jq,kq) = iN(0,ia,iq) * jN(0,ja,jq) * kN(1,ka,kq)
     end forall
     X(:,  iq,jq,kq) = (/ iX(iq),  jX(jq),  kX(kq) /)
     J(:,:,iq,jq,kq) = 0
     J(1,1,iq,jq,kq) = iJ
     J(2,2,iq,jq,kq) = jJ
     J(3,3,iq,jq,kq) = kJ
     detJ( iq,jq,kq) = iJ * jJ * kJ
  end forall

  forall (iq=1:inq, jq=1:jnq, kq=1:knq)
     forall (ia=1:ina, ja=1:jna, ka=1:kna)
        N2(d2(1,1),ia,ja,ka,iq,jq,kq) = iN(2,ia,iq) * jN(0,ja,jq) * kN(0,ka,kq)
        N2(d2(2,1),ia,ja,ka,iq,jq,kq) = iN(1,ia,iq) * jN(1,ja,jq) * kN(0,ka,kq)
        N2(d2(3,1),ia,ja,ka,iq,jq,kq) = iN(1,ia,iq) * jN(0,ja,jq) * kN(1,ka,kq)
        N2(d2(1,2),ia,ja,ka,iq,jq,kq) = iN(1,ia,iq) * jN(1,ja,jq) * kN(0,ka,kq)
        N2(d2(2,2),ia,ja,ka,iq,jq,kq) = iN(0,ia,iq) * jN(2,ja,jq) * kN(0,ka,kq)
        N2(d2(3,2),ia,ja,ka,iq,jq,kq) = iN(0,ia,iq) * jN(1,ja,jq) * kN(1,ka,kq)
        N2(d2(1,3),ia,ja,ka,iq,jq,kq) = iN(1,ia,iq) * jN(0,ja,jq) * kN(1,ka,kq)
        N2(d2(2,3),ia,ja,ka,iq,jq,kq) = iN(0,ia,iq) * jN(1,ja,jq) * kN(1,ka,kq)
        N2(d2(3,3),ia,ja,ka,iq,jq,kq) = iN(0,ia,iq) * jN(0,ja,jq) * kN(2,ka,kq)
     end forall
  end forall

end subroutine IGA_ShapeFuns_3D
