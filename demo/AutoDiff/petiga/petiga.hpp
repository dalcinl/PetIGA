#ifndef PETIGA_HPP
#define PETIGA_HPP

#include <petiga.h>

namespace PetIGA {

  template <int dim, int nen, int nsd=dim>
class Space {

  Space();
  Space(const Space&);
  Space& operator=(const Space&);

public:

  typedef const PetscReal (&Point)  [dim];
  typedef const PetscReal (&GeomMap)[nsd];
  static inline Point   GetPoint   (const IGAPoint q) { return reinterpret_cast<Point >(*q->mapU[0]); }
  static inline GeomMap GetGeomMap (const IGAPoint q) { return reinterpret_cast<Point >(*q->mapX[0]); }

  typedef const PetscReal (&PointU)[dim];
  typedef const PetscReal (&PointX)[nsd];
  typedef const PetscReal (&Normal)[nsd];
  static inline PointU GetPointU(const IGAPoint q) { return reinterpret_cast<Point >(*q->mapU[0]); }
  static inline PointX GetPointX(const IGAPoint q) { return reinterpret_cast<Point >(*q->mapX[0]); }
  static inline Normal GetNormal(const IGAPoint q) { return reinterpret_cast<Normal>(*q->normal ); }

  typedef const PetscReal (&Basis0)[nen];
  typedef const PetscReal (&Basis1)[nen][dim];
  typedef const PetscReal (&Basis2)[nen][dim][dim];
  typedef const PetscReal (&Basis3)[nen][dim][dim][dim];
  typedef const PetscReal (&Basis4)[nen][dim][dim][dim][dim];
  static inline Basis0 GetBasis0(const IGAPoint q) { return reinterpret_cast<Basis0>(*q->basis[0]); }
  static inline Basis1 GetBasis1(const IGAPoint q) { return reinterpret_cast<Basis1>(*q->basis[1]); }
  static inline Basis2 GetBasis2(const IGAPoint q) { return reinterpret_cast<Basis2>(*q->basis[2]); }
  static inline Basis3 GetBasis3(const IGAPoint q) { return reinterpret_cast<Basis3>(*q->basis[3]); }
  static inline Basis4 GetBasis4(const IGAPoint q) { return reinterpret_cast<Basis4>(*q->basis[4]); }

  typedef const PetscReal (&Shape0)[nen];
  typedef const PetscReal (&Shape1)[nen][dim];
  typedef const PetscReal (&Shape2)[nen][dim][dim];
  typedef const PetscReal (&Shape3)[nen][dim][dim][dim];
  typedef const PetscReal (&Shape4)[nen][dim][dim][dim][dim];
  static inline Shape0 GetShape0(const IGAPoint q) { return reinterpret_cast<Shape0>(*q->shape[0]); }
  static inline Shape1 GetShape1(const IGAPoint q) { return reinterpret_cast<Shape1>(*q->shape[1]); }
  static inline Shape2 GetShape2(const IGAPoint q) { return reinterpret_cast<Shape2>(*q->shape[2]); }
  static inline Shape3 GetShape3(const IGAPoint q) { return reinterpret_cast<Shape3>(*q->shape[3]); }
  static inline Shape4 GetShape4(const IGAPoint q) { return reinterpret_cast<Shape4>(*q->shape[4]); }

  template <int offset, int num, int dof, typename Scalar> static inline
  void Evaluate(const IGAPoint q, const Scalar (&U)[nen][dof], Scalar (&u)[num])
  {
    Shape0 N0 = GetShape0(q);
    for (int c=0; c<num; c++)
      u[c] = 0.0f;
    for (int a=0; a<nen; a++)
      for (int c=0; c<num; c++)
        u[c] += N0[a] * U[a][c+offset];
  }
  template <int offset, int num, int dof, typename Scalar> static inline
  void Evaluate(const IGAPoint q, const Scalar (&U)[nen][dof], Scalar (&u)[num][dim])
  {
    Shape1 N1 = GetShape1(q);
    for (int c=0; c<num; c++)
      for (int i=0; i<dim; i++)
        u[c][i] = 0.0f;
    for (int a=0; a<nen; a++)
      for (int c=0; c<num; c++)
        for (int i=0; i<dim; i++)
          u[c][i] += N1[a][i] * U[a][c+offset];
  }
  template <int offset, int num, int dof, typename Scalar> static inline
  void Evaluate(const IGAPoint q, const Scalar (&U)[nen][dof], Scalar (&u)[num][dim][dim])
  {
    Shape2 N2 = GetShape2(q);
    for (int c=0; c<num; c++)
      for (int i=0; i<dim; i++)
        for (int j=0; j<dim; j++)
          u[c][i][j] = 0.0f;
    for (int a=0; a<nen; a++)
      for (int c=0; c<num; c++)
        for (int i=0; i<dim; i++)
          for (int j=0; j<dim; j++)
            u[c][i][j] += N2[a][i][j] * U[a][c+offset];
  }
  template <int offset, int num, int dof, typename Scalar> static inline
  void Evaluate(const IGAPoint q, const Scalar (&U)[nen][dof], Scalar (&u)[num][dim][dim][dim])
  {
    Shape3 N3 = GetShape3(q);
    for (int c=0; c<num; c++)
      for (int i=0; i<dim; i++)
        for (int j=0; j<dim; j++)
          for (int k=0; k<dim; k++)
            u[c][i][j][k] = 0.0f;
    for (int a=0; a<nen; a++)
      for (int c=0; c<num; c++)
        for (int i=0; i<dim; i++)
          for (int j=0; j<dim; j++)
            for (int k=0; k<dim; k++)
              u[c][i][j][k] += N3[a][i][j][k] * U[a][c+offset];
  }
  template <int offset, int num, int dof, typename Scalar> static inline
  void Evaluate(const IGAPoint q, const Scalar (&U)[nen][dof], Scalar (&u)[num][dim][dim][dim][dim])
  {
    Shape4 N4 = GetShape4(q);
    for (int c=0; c<num; c++)
      for (int i=0; i<dim; i++)
        for (int j=0; j<dim; j++)
          for (int k=0; k<dim; k++)
            for (int l=0; l<dim; l++)
              u[c][i][j][k][l] = 0.0f;
    for (int a=0; a<nen; a++)
      for (int c=0; c<num; c++)
        for (int i=0; i<dim; i++)
          for (int j=0; j<dim; j++)
            for (int k=0; k<dim; k++)
              for (int l=0; l<dim; l++)
                u[c][i][j][k][l] += N4[a][i][j][k][l] * U[a][c+offset];
  }
  template <int offset, int dof, typename Scalar> static inline
  void EvaluateDiv(const IGAPoint q, const Scalar (&U)[nen][dof], Scalar &u)
  {
    Shape1 N1 = GetShape1(q);
    u = 0.0f;
    for (int a=0; a<nen; a++)
      for (int i=0; i<dim; i++)
        u += N1[a][i] * U[a][i+offset];
  }
  template <int offset, int num, int dof, typename Scalar> static inline
  void EvaluateDel2(const IGAPoint q, const Scalar (&U)[nen][dof], Scalar (&u)[num])
  {
    Shape2 N2 = GetShape2(q);
    for (int c=0; c<num; c++)
      u[c] = 0.0f;
    for (int a=0; a<nen; a++)
      for (int c=0; c<num; c++)
        for (int i=0; i<dim; i++)
          u[c] += N2[a][i][i] * U[a][c+offset];
  }

}; // class Space

namespace Point {

// ---

template <int i, int num, int nen, int dof, typename Scalar>
inline void FormValue (IGAPoint q, const Scalar (&U)[nen][dof], Scalar (&u)[num])
{
  typedef PetIGA::Space<1,nen> Space;
  Space::template Evaluate<i,num>(q, U, u);
}
template <int i, int num, int dim, int nen, int dof, typename Scalar>
inline void FormGrad(IGAPoint q, const Scalar (&U)[nen][dof], Scalar (&u)[num][dim])
{
  typedef PetIGA::Space<dim,nen> Space;
  Space::template Evaluate<i,num>(q, U, u);
}
template <int i, int num, int dim, int nen, int dof, typename Scalar>
inline void FormHess(IGAPoint q, const Scalar (&U)[nen][dof], Scalar (&u)[num][dim][dim])
{
  typedef PetIGA::Space<dim,nen> Space;
  Space::template Evaluate<i,num>(q, U, u);
}
template <int i, int num, int dim, int nen, int dof, typename Scalar>
inline void FormDer3(IGAPoint q, const Scalar (&U)[nen][dof], Scalar (&u)[num][dim][dim][dim])
{
  typedef PetIGA::Space<dim,nen> Space;
  Space::template Evaluate<i,num>(q, U, u);
}
template <int i, int num, int dim, int nen, int dof, typename Scalar>
inline void FormDer4(IGAPoint q, const Scalar (&U)[nen][dof], Scalar (&u)[num][dim][dim][dim][dim])
{
  typedef PetIGA::Space<dim,nen> Space;
  Space::template Evaluate<i,num>(q, U, u);
}

// ---

template <int nen, int dof, typename Scalar>
inline void FormValue(IGAPoint q, const Scalar (&U)[nen][dof], Scalar (&u)[dof])
{
  FormValue<0>(q,U,u);
}
template <int dim, int nen, int dof, typename Scalar>
inline void FormGrad(IGAPoint q, const Scalar (&U)[nen][dof], Scalar (&u)[dof][dim])
{
  FormGrad<0>(q,U,u);
}
template <int dim, int nen, int dof, typename Scalar>
inline void FormHess(IGAPoint q, const Scalar (&U)[nen][dof], Scalar (&u)[dof][dim][dim])
{
  FormHess<0>(q,U,u);
}
template <int dim, int nen, int dof, typename Scalar>
inline void FormDer3(IGAPoint q, const Scalar (&U)[nen][dof], Scalar (&u)[dof][dim][dim][dim])
{
  FormDer3<0>(q,U,u);
}
template <int dim, int nen, int dof, typename Scalar>
inline void FormDer4(IGAPoint q, const Scalar (&U)[nen][dof], Scalar (&u)[dof][dim][dim][dim][dim])
{
  FormDer4<0>(q,U,u);
}

// ---

template <int i, int nen, int dof, typename Scalar>
inline void FormValue(IGAPoint q, const Scalar (&U)[nen][dof], Scalar (&u))
{
  typedef Scalar (&array)[1];
  FormValue<i>(q, U, reinterpret_cast<array>(u));
}
template <int i, int dim, int nen, int dof, typename Scalar>
inline void FormGrad(IGAPoint q, const Scalar (&U)[nen][dof], Scalar (&u)[dim])
{
  typedef Scalar (&array)[1][dim];
  FormGrad<i>(q, U, reinterpret_cast<array>(u));
}
template <int i, int dim, int nen, int dof, typename Scalar>
inline void FormHess(IGAPoint q, const Scalar (&U)[nen][dof], Scalar (&u)[dim][dim])
{
  typedef Scalar (&array)[1][dim][dim];
  FormHess<i>(q, U, reinterpret_cast<array>(u));
}
template <int i, int dim, int nen, int dof, typename Scalar>
inline void FormDer3(IGAPoint q, const Scalar (&U)[nen][dof], Scalar (&u)[dim][dim][dim])
{
  typedef Scalar (&array)[1][dim][dim][dim];
  FormDer3<i>(q, U, reinterpret_cast<array>(u));
}
template <int i, int dim, int nen, int dof, typename Scalar>
inline void FormDer4(IGAPoint q, const Scalar (&U)[nen][dof], Scalar (&u)[dim][dim][dim][dim])
{
  typedef Scalar (&array)[1][dim][dim][dim][dim];
  FormDer4<i>(q, U, reinterpret_cast<array>(u));
}

// ---

template <int nen, typename Scalar>
inline void FormValue(IGAPoint q, const Scalar (&U)[nen][1], Scalar (&u))
{
  typedef Scalar (&array)[1];
  FormValue(q, U, reinterpret_cast<array>(u));
}
template <int dim, int nen, typename Scalar>
inline void FormGrad(IGAPoint q, const Scalar (&U)[nen][1], Scalar (&u)[dim])
{
  typedef Scalar (&array)[1][dim];
  FormGrad(q, U, reinterpret_cast<array>(u));
}
template <int dim, int nen, typename Scalar>
inline void FormHess(IGAPoint q, const Scalar (&U)[nen][1], Scalar (&u)[dim][dim])
{
  typedef Scalar (&array)[1][dim][dim];
  FormHess(q, U, reinterpret_cast<array>(u));
}
template <int dim, int nen, typename Scalar>
inline void FormDer3(IGAPoint q, const Scalar (&U)[nen][1], Scalar (&u)[dim][dim][dim])
{
  typedef Scalar (&array)[1][dim][dim][dim];
  FormDer3(q, U, reinterpret_cast<array>(u));
}
template <int dim, int nen, typename Scalar>
inline void FormDer4(IGAPoint q, const Scalar (&U)[nen][1], Scalar (&u)[dim][dim][dim][dim])
{
  typedef Scalar (&array)[1][dim][dim][dim][dim];
  FormDer4(q, U, reinterpret_cast<array>(u));
}

// ---

template <int i, int dim, int nen, int dof, typename Scalar>
inline void FormDiv(IGAPoint q, const Scalar (&U)[nen][dof], Scalar (&u))
{
  typedef PetIGA::Space<dim,nen> Space;
  Space::template EvaluateDiv<i>(q, U, u);
}
template <int dim, int nen, typename Scalar>
inline void FormDiv(IGAPoint q, const Scalar (&U)[nen][dim], Scalar (&u))
{
  FormDiv<0,dim,nen,dim>(q, U, u);
}

template <int i, int dim, int nen, int dof, int num, typename Scalar>
inline void FormDel2(IGAPoint q, const Scalar (&U)[nen][dof], Scalar (&u)[num])
{
  typedef PetIGA::Space<dim,nen> Space;
  Space::template EvaluateDel2<i,num>(q, U, u);
}
template <int dim, int nen, int dof, typename Scalar>
inline void FormDel2(IGAPoint q, const Scalar (&U)[nen][dof], Scalar (&u)[dof])
{
  FormDel2<0,dim>(q, U, u);
}
template <int i, int dim, int nen, int dof, typename Scalar>
inline void FormDel2(IGAPoint q, const Scalar (&U)[nen][dof], Scalar (&u))
{
  typedef Scalar (&array)[1];
  FormDel2<i,dim>(q, U, reinterpret_cast<array>(u));
}
template <int dim, int nen, typename Scalar>
inline void FormDel2(IGAPoint q, const Scalar (&U)[nen][1], Scalar (&u))
{
  FormDel2<0,dim>(q, U, u);
}

// ---

} // namespace Point

template <int dim, typename ScalarA, typename ScalarB>
inline ScalarB dot(const ScalarA (&a)[dim], const ScalarB (&b)[dim])
{
  ScalarB s = 0.0f;
  for (int i=0; i<dim; i++)
    s += a[i]*b[i];
  return s;
}

template <typename ScalarA, typename ScalarB>
inline ScalarB dot(const ScalarA (&a)[3], const ScalarB (&b)[3])
{ return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }

template <typename ScalarA, typename ScalarB>
inline ScalarB dot(const ScalarA (&a)[2], const ScalarB (&b)[2])
{ return a[0]*b[0] + a[1]*b[1]; }

template <typename ScalarA, typename ScalarB>
inline ScalarB dot(const ScalarA (&a)[1], const ScalarB (&b)[1])
{ return a[0]*b[0]; }

template <int dim, typename Scalar>
inline Scalar trace(const Scalar (&a)[dim][dim])
{
  Scalar s = 0.0f;
  for (int i=0; i<dim; i++)
    s += a[i][i];
  return s;
}

template <typename Scalar>
inline Scalar trace(const Scalar (&a)[3][3])
{ return a[0][0] + a[1][1] + a[2][2]; }

template <typename Scalar>
inline Scalar trace(const Scalar (&a)[2][2])
{ return a[0][0] + a[1][1]; }

template <typename Scalar>
inline Scalar trace(const Scalar (&a)[1][1])
{ return a[0][0]; }

template <int dim, typename ScalarA, typename ScalarB, typename ScalarC>
inline void outer(const ScalarA (&a)[dim],
                  const ScalarB (&b)[dim],
                  /* */ ScalarC (&c)[dim][dim])
{
  for (int i=0; i<dim; i++)
    for (int j=0; j<dim; j++)
      c[i][j] = a[i] * b[j];
}

template <int dim, typename ScalarW, typename ScalarU>
inline void adv(const ScalarW (&w)[dim],
                const ScalarU (&grd_u)[dim][dim],
                /* */ ScalarU (&adv_u)[dim])
{
  for (int i=0; i<dim; i++)
    adv_u[i] = dot(w, grd_u[i]);
}

template <int dim, typename Scalar>
inline Scalar div(const Scalar (&grd_u)[dim][dim])
{ return trace(grd_u); }

template <int dim, typename Scalar>
inline Scalar del2(const Scalar (&a)[dim][dim])
{ return trace(a); }

} // namespace PetIGA

#endif
