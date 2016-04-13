pure subroutine IGA_Quadrature_1D(&
     inq,iX,iW,iJ,                &
     X,W,J)                       &
  bind(C, name="IGA_Quadrature_1D")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), parameter        :: dim = 1
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: inq
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: iX(inq), iW(inq), iJ
  real   (kind=IGA_REAL_KIND   ), intent(out) :: X(dim,inq)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: W(    inq)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: J(    inq)
  integer(kind=IGA_INTEGER_KIND)  :: iq
  do iq=1,inq
     X(1,iq) = iX(iq)
     W(  iq) = iW(iq)
  end do
  J = iJ
end subroutine IGA_Quadrature_1D


pure subroutine IGA_BasisFuns_1D(&
     order,                      &
     inq,ina,iN,                 &
     N0,N1,N2,N3,N4)             &
  bind(C, name="IGA_BasisFuns_1D")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), parameter        :: dim = 1
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: order
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: inq, ina
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: iN(0:4,ina,inq)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N0(dim**0,ina,inq)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N1(dim**1,ina,inq)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N2(dim**2,ina,inq)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N3(dim**3,ina,inq)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N4(dim**4,ina,inq)
  integer(kind=IGA_INTEGER_KIND)  :: iq
  do iq=1,inq
     call TensorBasisFuns(&
          order,&
          ina,iN(:,:,iq),&
          N0(:,:,iq),&
          N1(:,:,iq),&
          N2(:,:,iq),&
          N3(:,:,iq),&
          N4(:,:,iq))
  end do
contains
pure subroutine TensorBasisFuns(&
     order,&
     ina,iN,&
     N0,N1,N2,N3,N4)
  implicit none
  integer(kind=IGA_INTEGER_KIND), parameter        :: dim = 1
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: order
  integer(kind=IGA_INTEGER_KIND), intent(in),value :: ina
  real   (kind=IGA_REAL_KIND   ), intent(in)  :: iN(0:4,ina)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N0(                ina)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N1(            dim,ina)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N2(        dim,dim,ina)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N3(    dim,dim,dim,ina)
  real   (kind=IGA_REAL_KIND   ), intent(out) :: N4(dim,dim,dim,dim,ina)
  integer(kind=IGA_INTEGER_KIND)  :: ia
  !
  do ia=1,ina
     N0(ia) = iN(0,ia)
  end do
  !
  if (order < 1) return
  do ia=1,ina
     N1(1,ia) = iN(1,ia)
  end do
  !
  if (order < 2) return
  do ia=1,ina
     N2(1,1,ia) = iN(2,ia)
  end do
  !
  if (order < 3) return
  do ia=1,ina
     N3(1,1,1,ia) = iN(3,ia)
  end do
  !
  if (order < 4) return
  do ia=1,ina
     N4(1,1,1,1,ia) = iN(4,ia)
  end do
  !
end subroutine TensorBasisFuns
end subroutine IGA_BasisFuns_1D


pure subroutine IGA_Rationalize_1D(&
     order,                      &
     nqp,nen,W,                  &
     N0,N1,N2,N3,N4)             &
  bind(C, name="IGA_Rationalize_1D")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), parameter        :: dim = 1
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
end subroutine IGA_Rationalize_1D


pure subroutine IGA_GeometryMap_1D(&
     order,                        &
     nqp,nen,X,                    &
     M0,M1,M2,M3,M4,               &
     X0,X1,X2,X3,X4)               &
  bind(C, name="IGA_GeometryMap_1D")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), parameter        :: dim = 1
  integer(kind=IGA_INTEGER_KIND), parameter        :: nsd = 1
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
end subroutine IGA_GeometryMap_1D


pure subroutine IGA_InverseMap_1D(&
     order,                       &
     nqp,                         &
     X1,X2,X3,X4,                 &
     dX,                          &
     E1,E2,E3,E4)                 &
  bind(C, name="IGA_InverseMap_1D")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), parameter        :: dim = 1
  integer(kind=IGA_INTEGER_KIND), parameter        :: nsd = 1
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
end subroutine IGA_InverseMap_1D


pure subroutine IGA_ShapeFuns_1D(&
     order,                      &
     nqp,nen,                    &
     E1,E2,E3,E4,                &
     M1,M2,M3,M4,                &
     N1,N2,N3,N4)                &
  bind(C, name="IGA_ShapeFuns_1D")
  use PetIGA
  implicit none
  integer(kind=IGA_INTEGER_KIND), parameter        :: dim = 1
  integer(kind=IGA_INTEGER_KIND), parameter        :: nsd = 1
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
end subroutine IGA_ShapeFuns_1D
