subroutine IGA_Quadrature_2D(&
     inq,iX,iW, &
     jnq,jX,jW, &
     X, W)      &
  bind(C, name="IGA_Quadrature_2D")
  use ISO_C_BINDING, only: C_INT, C_LONG
  use ISO_C_BINDING, only: C_FLOAT, C_DOUBLE
  implicit none
  integer, parameter :: dim = 2
  integer(kind=C_INT   ), intent(in),value :: inq
  integer(kind=C_INT   ), intent(in),value :: jnq
  real   (kind=C_DOUBLE), intent(in)  :: iX(inq), iW(inq)
  real   (kind=C_DOUBLE), intent(in)  :: jX(jnq), jW(jnq)
  real   (kind=C_DOUBLE), intent(out) :: X(dim,inq,jnq)
  real   (kind=C_DOUBLE), intent(out) :: W(inq,jnq)
  integer :: iq
  integer :: jq
  forall (iq=1:inq, jq=1:jnq)
     X(:,iq,jq) = (/ iX(iq),  jX(jq)/)
     W(  iq,jq) =    iW(iq) * jW(jq)
  end forall
end subroutine IGA_Quadrature_2D

subroutine IGA_ShapeFuns_2D(&
     inq,ina,ind,iX,iJ,iN, &
     jnq,jna,jnd,jX,jJ,jN, &
     detJ,X,J,N0,N1,N2,N3) &
  bind(C, name="IGA_ShapeFuns_2D")
  use ISO_C_BINDING, only: C_INT, C_LONG
  use ISO_C_BINDING, only: C_FLOAT, C_DOUBLE
  implicit none
  integer(kind=C_INT   ), parameter        :: dim = 2
  integer(kind=C_INT   ), intent(in),value :: inq, ina, ind
  integer(kind=C_INT   ), intent(in),value :: jnq, jna, jnd
  real   (kind=C_DOUBLE), intent(in)  :: iX(inq), iJ, iN(0:ind,ina,inq)
  real   (kind=C_DOUBLE), intent(in)  :: jX(jnq), jJ, jN(0:jnd,jna,jnq)
  real   (kind=C_DOUBLE), intent(out) :: detJ(     inq,jnq)
  real   (kind=C_DOUBLE), intent(out) :: X(    dim,inq,jnq)
  real   (kind=C_DOUBLE), intent(out) :: J(dim,dim,inq,jnq)
  real   (kind=C_DOUBLE), intent(out) :: N0(            ina,jna,inq,jnq)
  real   (kind=C_DOUBLE), intent(out) :: N1(        dim,ina,jna,inq,jnq)
  real   (kind=C_DOUBLE), intent(out) :: N2(    dim,dim,ina,jna,inq,jnq)
  real   (kind=C_DOUBLE), intent(out) :: N3(dim,dim,dim,ina,jna,inq,jnq)
  integer :: ia,iq
  integer :: ja,jq

  forall (iq=1:inq, jq=1:jnq)
     forall (ia=1:ina, ja=1:jna)
        !
        N0(ia,ja,iq,jq)   = iN(0,ia,iq) * jN(0,ja,jq)
        !
        N1(1,ia,ja,iq,jq) = iN(1,ia,iq) * jN(0,ja,jq)
        N1(2,ia,ja,iq,jq) = iN(0,ia,iq) * jN(1,ja,jq)
     end forall
     X(:,  iq,jq) = (/ iX(iq),  jX(jq) /)
     J(:,:,iq,jq) = 0
     J(1,1,iq,jq) = iJ
     J(2,2,iq,jq) = jJ
     detJ( iq,jq) = iJ * jJ
  end forall

  forall (iq=1:inq, jq=1:jnq)
     forall (ia=1:ina, ja=1:jna)
        N2(1,1,ia,ja,iq,jq) = iN(2,ia,iq) * jN(0,ja,jq)
        N2(2,1,ia,ja,iq,jq) = iN(1,ia,iq) * jN(1,ja,jq)
        N2(1,2,ia,ja,iq,jq) = iN(1,ia,iq) * jN(1,ja,jq)
        N2(2,2,ia,ja,iq,jq) = iN(0,ia,iq) * jN(2,ja,jq)
     end forall
  end forall

end subroutine IGA_ShapeFuns_2D
