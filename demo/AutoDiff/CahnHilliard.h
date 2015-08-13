#include <petiga.h>

typedef struct {
  PetscReal theta;
  PetscReal alpha;
  PetscReal cbar;
} Params;

static inline void Mobility(const Params *params,PetscScalar c,PetscScalar *M,PetscScalar *dM,PetscScalar *d2M)
{
  (void)params; /* unused */
  if (M  ) *M   = c*(1-c);
  if (dM ) *dM  = 1-2*c;
  if (d2M) *d2M = -2;
}

static inline void ChemicalPotential(const Params *params,PetscScalar c,PetscScalar *mu,PetscScalar *dmu,PetscScalar *d2mu)
{
  PetscReal theta  = params->theta;
  PetscReal alpha  = params->alpha;
  if (mu  ) *mu   = 3*alpha * (0.5/theta*log(c/(1-c)) + 1 - 2*c);
  if (dmu ) *dmu  = 3*alpha * (0.5/theta*1/(c*(1-c)) - 2);
  if (d2mu) *d2mu = 3*alpha * (0.5/theta*(2*c-1)/(c*c*(1-c)*(1-c)));
}

static inline
PetscScalar dots(PetscInt dim,const PetscReal a[],const PetscScalar b[])
{
  PetscInt i;
  PetscScalar s = 0;
  for (i=0; i<dim; i++) s += a[i]*b[i];
  return s;
}

static inline
PetscReal dotr(PetscInt dim,const PetscReal a[],const PetscReal b[])
{
  PetscInt i;
  PetscReal s = 0;
  for (i=0; i<dim; i++) s += a[i]*b[i];
  return s;
}

static inline
PetscReal del2(PetscInt dim,const PetscReal a[dim][dim])
{
  PetscInt i;
  PetscReal s = 0;
  for (i=0; i<dim; i++) s += a[i][i];
  return s;
}
