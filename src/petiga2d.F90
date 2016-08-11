pure subroutine IGA_Quadrature_2D(&
     inq,iX,iW,iJ,                &
     jnq,jX,jW,jJ,                &
     X,W,J)                       &
  bind(C, name="IGA_Quadrature_2D")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), parameter        :: dim = 2
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: inq
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: jnq
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: iX(inq), iW(inq), iJ
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: jX(jnq), jW(jnq), jJ
  real   (kind=IGA_REAL_KIND   ), intent(out) :: X(dim,inq,jnq)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: W(    inq,jnq)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: J(    inq,jnq)
  integer(kind=IGA_INTEGER_KIND)  :: iq
  integer(kind=IGA_INTEGER_KIND)  :: jq
  do jq=1,jnq; do iq=1,inq
     X(1,iq,jq) = iX(iq)
     X(2,iq,jq) = jX(jq)
     W(  iq,jq) = iW(iq) * jW(jq)
  end do; end do
  J = iJ * jJ
end subroutine IGA_Quadrature_2D


pure subroutine IGA_BasisFuns_2D(&
     order,                      &
     inq,ina,iN,                 &
     jnq,jna,jN,                 &
     N0,N1,N2,N3,N4)             &
  bind(C, name="IGA_BasisFuns_2D")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), parameter        :: dim = 2
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: order
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: inq, ina
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: jnq, jna
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: iN(0:4,ina,inq)
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: jN(0:4,jna,jnq)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N0(dim**0,ina*jna,inq,jnq)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N1(dim**1,ina*jna,inq,jnq)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N2(dim**2,ina*jna,inq,jnq)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N3(dim**3,ina*jna,inq,jnq)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N4(dim**4,ina*jna,inq,jnq)
  integer(kind=IGA_INTEGER_KIND)  :: iq
  integer(kind=IGA_INTEGER_KIND)  :: jq
  do jq=1,jnq; do iq=1,inq
     call TensorBasisFuns(&
          order,&
          ina,iN(:,:,iq),&
          jna,jN(:,:,jq),&
          N0(:,:,iq,jq),&
          N1(:,:,iq,jq),&
          N2(:,:,iq,jq),&
          N3(:,:,iq,jq),&
          N4(:,:,iq,jq))
  end do; end do
contains
pure subroutine TensorBasisFuns(&
     order,&
     ina,iN,&
     jna,jN,&
     N0,N1,N2,N3,N4)
  implicit none
  integer(kind=IGA_INTEGER_KIND), parameter        :: dim = 2
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: order
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: ina
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: jna
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: iN(0:4,ina)
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: jN(0:4,jna)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N0(                ina,jna)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N1(            dim,ina,jna)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N2(        dim,dim,ina,jna)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N3(    dim,dim,dim,ina,jna)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N4(dim,dim,dim,dim,ina,jna)
  integer(kind=IGA_INTEGER_KIND)  :: ia, ja
  !
   do ja=1,jna; do ia=1,ina
     N0(ia,ja) = iN(0,ia) * jN(0,ja)
  end do; end do
  !
  if (order < 1) return
  do ja=1,jna; do ia=1,ina
     N1(1,ia,ja) = iN(1,ia) * jN(0,ja)
     N1(2,ia,ja) = iN(0,ia) * jN(1,ja)
  end do; end do
  !
  if (order < 2) return ! XXX Optimize!
  do ja=1,jna; do ia=1,ina
     N2(1,1,ia,ja) = iN(2,ia) * jN(0,ja)
     N2(2,1,ia,ja) = iN(1,ia) * jN(1,ja)
     N2(1,2,ia,ja) = iN(1,ia) * jN(1,ja)
     N2(2,2,ia,ja) = iN(0,ia) * jN(2,ja)
  end do; end do
  !
  if (order < 3) return ! XXX Optimize!
  do ja=1,jna; do ia=1,ina
     N3(1,1,1,ia,ja) = iN(3,ia) * jN(0,ja)
     N3(2,1,1,ia,ja) = iN(2,ia) * jN(1,ja)
     N3(1,2,1,ia,ja) = iN(2,ia) * jN(1,ja)
     N3(2,2,1,ia,ja) = iN(1,ia) * jN(2,ja)
     N3(1,1,2,ia,ja) = iN(2,ia) * jN(1,ja)
     N3(2,1,2,ia,ja) = iN(1,ia) * jN(2,ja)
     N3(1,2,2,ia,ja) = iN(1,ia) * jN(2,ja)
     N3(2,2,2,ia,ja) = iN(0,ia) * jN(3,ja)
  end do; end do
  !
  if (order < 4) return ! XXX Optimize!
  do ja=1,jna; do ia=1,ina
     N4(1,1,1,1,ia,ja) = iN(4,ia) * jN(0,ja)
     N4(2,1,1,1,ia,ja) = iN(3,ia) * jN(1,ja)
     N4(1,2,1,1,ia,ja) = iN(3,ia) * jN(1,ja)
     N4(2,2,1,1,ia,ja) = iN(2,ia) * jN(2,ja)
     N4(1,1,2,1,ia,ja) = iN(3,ia) * jN(1,ja)
     N4(2,1,2,1,ia,ja) = iN(2,ia) * jN(2,ja)
     N4(1,2,2,1,ia,ja) = iN(2,ia) * jN(2,ja)
     N4(2,2,2,1,ia,ja) = iN(1,ia) * jN(3,ja)
     N4(1,1,1,2,ia,ja) = iN(3,ia) * jN(1,ja)
     N4(2,1,1,2,ia,ja) = iN(2,ia) * jN(2,ja)
     N4(1,2,1,2,ia,ja) = iN(2,ia) * jN(2,ja)
     N4(2,2,1,2,ia,ja) = iN(1,ia) * jN(3,ja)
     N4(1,1,2,2,ia,ja) = iN(2,ia) * jN(2,ja)
     N4(2,1,2,2,ia,ja) = iN(1,ia) * jN(3,ja)
     N4(1,2,2,2,ia,ja) = iN(1,ia) * jN(3,ja)
     N4(2,2,2,2,ia,ja) = iN(0,ia) * jN(4,ja)
  end do; end do
  !
end subroutine TensorBasisFuns
end subroutine IGA_BasisFuns_2D


pure subroutine IGA_Rationalize_2D(&
     order,                      &
     nqp,nen,W,                  &
     N0,N1,N2,N3,N4)             &
  bind(C, name="IGA_Rationalize_2D")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), parameter        :: dim = 2
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
end subroutine IGA_Rationalize_2D


pure subroutine IGA_GeometryMap_2D(&
     order,                        &
     nqp,nen,X,                    &
     M0,M1,M2,M3,M4,               &
     X0,X1,X2,X3,X4)               &
  bind(C, name="IGA_GeometryMap_2D")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), parameter        :: dim = 2
  integer(kind=IGA_INTEGER_KIND), parameter        :: nsd = 2
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
end subroutine IGA_GeometryMap_2D


pure subroutine IGA_InverseMap_2D(&
     order,                       &
     nqp,                         &
     X1,X2,X3,X4,                 &
     dX,                          &
     E1,E2,E3,E4)                 &
  bind(C, name="IGA_InverseMap_2D")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), parameter        :: dim = 2
  integer(kind=IGA_INTEGER_KIND), parameter        :: nsd = 2
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
end subroutine IGA_InverseMap_2D


pure subroutine IGA_ShapeFuns_2D(&
     order,                      &
     nqp,nen,                    &
     E1,E2,E3,E4,                &
     M1,M2,M3,M4,                &
     N1,N2,N3,N4)                &
  bind(C, name="IGA_ShapeFuns_2D")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), parameter        :: dim = 2
  integer(kind=IGA_INTEGER_KIND), parameter        :: nsd = 2
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
end subroutine IGA_ShapeFuns_2D


pure subroutine IGA_BoundaryArea_2D(&
     m,axis,side,              &
     geometry,Cx,              &
     rational,Cw,              &
     nqp,W,nen,N,              &
     dS)                       &
  bind(C, name="IGA_BoundaryArea_2D")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), parameter         :: nsd = 2
  integer(kind=IGA_INTEGER_KIND), parameter         :: dim = 1
  integer(kind=IGA_INTEGER_KIND), intent(in)        :: m(2)
  integer(kind=IGA_INTEGER_KIND), intent(in),value  :: axis, side
  integer(kind=IGA_INTEGER_KIND), intent(in),value  :: geometry, rational
  real   (kind=IGA_REAL_KIND   ), intent(in)        :: Cx(nsd,m(1),m(2))
  real   (kind=IGA_REAL_KIND   ), intent(in)        :: Cw(    m(1),m(2))
  integer(kind=IGA_INTEGER_KIND), intent(in),value  :: nqp, nen
  real   (kind=IGA_REAL_KIND   ), intent(in)        :: W(nqp), N(0:4,nen,nqp)
  real   (kind=IGA_REAL_KIND   ), intent(out)       :: dS
  integer(kind=IGA_INTEGER_KIND)  :: k, q
  real   (kind=IGA_REAL_KIND   )  :: N0(nen), N1(dim,nen), detJ
  real   (kind=IGA_REAL_KIND   )  :: Xw(nen), Xx(nsd,nen)
  select case (axis)
  case (0)
     if (side==0) k=1
     if (side==1) k=m(1)
     Xx = Cx(:,k,:); Xw = Cw(k,:)
  case (1)
     if (side==0) k=1
     if (side==1) k=m(2)
     Xx = Cx(:,:,k); Xw = Cw(:,k)
  end select
  detJ = 1
  dS = 0
  do q=1,nqp
     N0(  :) = N(0,:,q)
     N1(1,:) = N(1,:,q)
     if (rational /= 0) call Rationalize(nen,Xw,N0,N1)
     if (geometry /= 0) call Jacobian(nen,N1,Xx,detJ)
     dS = dS + detJ * W(q)
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
end subroutine IGA_BoundaryArea_2D
