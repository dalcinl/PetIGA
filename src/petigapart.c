#include <petsc.h>
#include <petsc/private/petscimpl.h>

PETSC_EXTERN PetscErrorCode IGA_Partition(PetscInt,PetscInt,PetscInt,const PetscInt[],PetscInt[],PetscInt[]);
PETSC_EXTERN PetscErrorCode IGA_Distribute(PetscInt,const PetscInt[],const PetscInt[],const PetscInt[],PetscInt[],PetscInt[]);

PETSC_STATIC_INLINE
PetscInt IGA_CUT2D(PetscInt M,PetscInt N,
                   PetscInt m,PetscInt n)
{return M*(n-1) + N*(m-1);}

PETSC_STATIC_INLINE
PetscInt IGA_CUT3D(PetscInt M,PetscInt N,PetscInt P,
                   PetscInt m,PetscInt n,PetscInt p)
{return (N*P*(m-1) + M*P*(n-1) + M*N*(p-1));}

PETSC_STATIC_INLINE
PetscInt IGA_PART2D_INNER(PetscInt size,
                          PetscInt   M,PetscInt   N,
                          PetscInt *_m,PetscInt *_n)
{
  PetscInt m,n;
  m = (PetscInt)(0.5 + sqrt(((double)M)/((double)N)*((double)size)));
  if (m == 0) {m = 1;} while (m > 0 && size % m) m--;
  n = size / m;
  *_m = m; *_n = n;
  return IGA_CUT2D(M,N,m,n);
}

PETSC_STATIC_INLINE
void IGA_PART2D(PetscInt size,
                PetscInt   M,PetscInt   N,
                PetscInt *_m,PetscInt *_n)
{
  PetscInt m,n;
  PetscInt m1,n1,a;
  PetscInt m2,n2,b;
  a = IGA_PART2D_INNER(size,M,N,&m1,&n1);
  b = IGA_PART2D_INNER(size,N,M,&n2,&m2);
  if (a<b) {m = m1; n = n1;}
  else     {m = m2; n = n2;}
  if (M == N && n < m) {PetscInt t = m; m = n; n = t;}
  *_m = m; *_n = n;
}

PETSC_STATIC_INLINE
PetscInt IGA_PART3D_INNER(PetscInt size,
                          PetscInt   M,PetscInt   N,PetscInt   P,
                          PetscInt *_m,PetscInt *_n,PetscInt *_p)
{
  PetscInt m,n,p,C;
  PetscInt mm,nn,pp,CC;
  /**/
  m = (PetscInt)(0.5 + pow(((double)M*(double)M)/((double)N*(double)P)*(double)size,1./3.));
  if (m == 0) {m = 1;} while (m > 0 && size % m) m--;
  /**/
  IGA_PART2D(size/m,N,P,&n,&p);
  C = IGA_CUT3D(M,N,P,m,n,p);
  /**/
  for (mm=m; mm>=1; mm--) {
    if (size % mm) continue;
    IGA_PART2D(size/mm,N,P,&nn,&pp);
    CC = IGA_CUT3D(M,N,P,mm,nn,pp);
    if (CC < C) {m = mm; n = nn; p = pp; C = CC;}
  }
  /**/
  for (nn=n; nn>=1; nn--) {
    if (size % nn) continue;
    IGA_PART2D(size/nn,M,P,&mm,&pp);
    CC = IGA_CUT3D(M,N,P,mm,nn,pp);
    if (CC < C) {m = mm; n = nn; p = pp; C = CC;}
  }
  /**/
  for (pp=p; pp>=1; pp--) {
    if (size % pp) continue;
    IGA_PART2D(size/pp,M,N,&mm,&nn);
    CC = IGA_CUT3D(M,N,P,mm,nn,pp);
    if (CC < C) {m = mm; n = nn; p = pp; C = CC;}
  }
  /**/
  *_m = m; *_n = n; *_p = p;
  return IGA_CUT3D(M,N,P,m,n,p);
}

PETSC_STATIC_INLINE
void IGA_PART3D(PetscInt size,
                PetscInt   M,PetscInt   N,PetscInt   P,
                PetscInt *_m,PetscInt *_n,PetscInt *_p)
{
  PetscInt m[3],n[3],p[3],C[3],k,i=0,Cmin=PETSC_MAX_INT,t;
  C[0] = IGA_PART3D_INNER(size,M,N,P,&m[0],&n[0],&p[0]);
  C[1] = IGA_PART3D_INNER(size,N,M,P,&n[1],&m[1],&p[1]);
  C[2] = IGA_PART3D_INNER(size,P,M,N,&p[2],&m[2],&n[2]);
  for (k=0; k<3; k++) if (C[k]<Cmin) {Cmin=C[k]; i=k;}
  if (M == N && n[i] < m[i]) {t = m[i]; m[i] = n[i]; n[i] = t;}
  if (M == P && p[i] < m[i]) {t = m[i]; m[i] = p[i]; p[i] = t;}
  if (N == P && p[i] < n[i]) {t = n[i]; n[i] = p[i]; p[i] = t;}
  *_m = m[i]; *_n = n[i]; *_p = p[i];
}

PETSC_STATIC_INLINE
void IGA_Part2D(PetscInt size,
                PetscInt   M,PetscInt   N,
                PetscInt *_m,PetscInt *_n)
{
  PetscInt m,n;
  m = *_m; n = *_n;
  if (m < 1 && n < 1) IGA_PART2D(size,M,N,&m,&n);
  else if (m < 1)     m = size / n;
  else if (n < 1)     n = size / m;
  *_m = m; *_n = n;
}

PETSC_STATIC_INLINE
void IGA_Part3D(PetscInt size,
                PetscInt   M,PetscInt   N,PetscInt   P,
                PetscInt *_m,PetscInt *_n,PetscInt *_p)
{
  PetscInt m,n,p;
  m = *_m; n = *_n; p = *_p;
  if (m < 1 && n < 1 && p < 1) IGA_PART3D(size,M,N,P,&m,&n,&p);
  else if (m < 1 && n < 1)     IGA_PART2D(size/p,M,N,&m,&n);
  else if (m < 1 && p < 1)     IGA_PART2D(size/n,M,P,&m,&p);
  else if (n < 1 && p < 1)     IGA_PART2D(size/m,N,P,&n,&p);
  else if (m < 1)              m = size / (n*p);
  else if (n < 1)              n = size / (m*p);
  else if (p < 1)              p = size / (m*n);
  *_m = m; *_n = n; *_p = p;
}

PetscErrorCode IGA_Partition(PetscInt size,PetscInt rank,
                             PetscInt dim,const PetscInt N[],
                             PetscInt n[],PetscInt i[])
{
  PetscInt k,p=1,m[3]={1,1,1};
  PetscFunctionBegin;
  PetscValidIntPointer(N,4);
  PetscValidIntPointer(n,5);
  if (i) PetscValidIntPointer(i,6);
  if (size < 1)
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
             "Number of partitions %D must be positive",size);
  if (i && (rank < 0 || rank >= size))
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
             "Partition index %D must be in range [0,%D]",rank,size-1);

  switch (dim) {
  case 3:  IGA_Part3D(size,N[0],N[1],N[2],&n[0],&n[1],&n[2]); break;
  case 2:  IGA_Part2D(size,N[0],N[1],&n[0],&n[1]); break;
  case 1:  if (n[0] < 1) n[0] = size; break;
  default: SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
                    "Number of dimensions %D must be in range [1,3]",dim);
  }
  for (k=0; k<dim; k++) p *= (m[k] = n[k]);
  if (p != size) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
                          "Bad partition, prod(%D,%D,%D) != %D",m[0],m[1],m[2],size);
  for (k=0; k<dim; k++)
    if (N[k] < n[k]) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
                              "Partition %D is too fine, %D elements in %D parts",k,N[k],n[k]);
  if (i)
    for (k=0; k<dim; k++) {
      i[k] = rank % n[k];
      rank -= i[k];
      rank /= n[k];
    }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE
void IGA_Dist1D(PetscInt size,PetscInt rank,
                PetscInt N,PetscInt *n,PetscInt *s)
{
  *n = N/size + ((N % size) > rank);
  *s = rank * (N/size) + (((N % size) > rank) ? rank : (N % size));
}

PetscErrorCode IGA_Distribute(PetscInt dim,
                              const PetscInt size[],const PetscInt rank[],
                              const PetscInt N[],PetscInt n[],PetscInt s[])
{
  PetscInt k;
  PetscFunctionBegin;
  PetscValidIntPointer(size,2);
  PetscValidIntPointer(rank,3);
  PetscValidIntPointer(N,4);
  PetscValidIntPointer(n,5);
  PetscValidIntPointer(s,6);
  if (dim < 1 || dim > 3)
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
             "Number of dimensions %D must be in range [1,3]",dim);
  for (k=0; k<dim; k++) {
    if (size[k] < 1)
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
               "Number of partitions %D must be positive",size[k]);
    if (rank[k] < 0 || rank[k] >= size[k])
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
               "Partition index %D must be in range [0,%D]",rank[k],size[k]-1);
    if (N[k] < 0)
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,
               "Number of elements %D must be non-negative",N[k]);
  }
  for (k=0; k<dim; k++)
    IGA_Dist1D(size[k],rank[k],N[k],&n[k],&s[k]);
  PetscFunctionReturn(0);
}
