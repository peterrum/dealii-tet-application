// ---------------------------------------------------------------------
//
// Copyright (C) 2020 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------


// Solve Poisson problem on a tet mesh and on a quad mesh with the same number
// of subdivisions.

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_isoparametric.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/simplex/fe_lib.h>
#include <deal.II/simplex/grid_generator.h>

using namespace dealii;

template <int dim>
class RightHandSideFunction : public Function<dim>
{
public:
  RightHandSideFunction(const unsigned int n_components)
    : Function<dim>(n_components)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const
  {
    return p[component % dim] * p[component % dim];
  }
};

template <int dim, int spacedim = dim>
void
test(const FiniteElement<dim, spacedim> &fe, const unsigned int n_components)
{
  Triangulation<dim, spacedim> tria;
  Simplex::GridGenerator::subdivided_hyper_cube(tria, 2);

  DoFHandler<dim> dof_handler(tria);

  dof_handler.distribute_dofs(fe);

  Vector<double> solution(dof_handler.n_dofs());

  MappingIsoparametric<dim> mapping(Simplex::FE_P<dim>(1));

  VectorTools::interpolate(mapping,
                           dof_handler,
                           RightHandSideFunction<dim>(n_components),
                           solution);

  DataOut<dim> data_out;

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");


  data_out.build_patches(mapping, /*1 or*/ 2);

  std::ofstream output("test.vtk");
  data_out.write_vtk(output);
}

int
main()
{
  {
    const unsigned int dim = 2;
    test<dim>(Simplex::FE_P<dim>(2) /*=degree*/, 1);
    test<dim>(FESystem<dim>(Simplex::FE_P<dim>(2 /*=degree*/), dim), dim);
    test<dim>(FESystem<dim>(Simplex::FE_P<dim>(2 /*=degree*/),
                            dim,
                            Simplex::FE_P<dim>(1 /*=degree*/),
                            1),
              dim + 1);
  }
  {
    const unsigned int dim = 3;
    test<dim>(Simplex::FE_P<dim>(2) /*=degree*/, 1);
    test<dim>(FESystem<dim>(Simplex::FE_P<dim>(2 /*=degree*/), dim), dim);
    test<dim>(FESystem<dim>(Simplex::FE_P<dim>(2 /*=degree*/),
                            dim,
                            Simplex::FE_P<dim>(1 /*=degree*/),
                            1),
              dim + 1);
  }
}
