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

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_isoparametric.h>

#include <deal.II/grid/tria.h>

#include <deal.II/tet/fe_q.h>
#include <deal.II/tet/grid_generator.h>
#include <deal.II/tet/quadrature_lib.h>

using namespace dealii;

template <int dim, int spacedim = dim>
void
test()
{
  const unsigned int degree = 2;

  // create triangulation
  Triangulation<dim, spacedim> tria;
  std::vector<unsigned int>    sub{2, 2, 2};
  Point<dim>                   p1(0, 0, 0);
  Point<dim>                   p2(1, 1, 1);
  Tet::GridGenerator::subdivided_hyper_rectangle(tria, sub, p1, p2, false);

  // mapping
  Tet::FE_Q<dim>            fe_mapping(1);
  MappingIsoparametric<dim> mapping(fe_mapping);

  // finite element
  Tet::FE_Q<dim> fe(degree);

  // quadrature rule
  Tet::QGauss<dim - 1> quad(dim == 2 ? (degree == 1 ? 2 : 3) : // TODO
                              (degree == 1 ? 3 : 7));

  // create FEFaceValues
  const UpdateFlags flag = update_JxW_values | update_values | update_gradients;
  FEFaceValues<dim, spacedim> fe_face_values(mapping, fe, quad, flag);

  for (const auto &cell : tria.active_cell_iterators())
    for (const auto face_index : cell->face_indices())
      {
        fe_face_values.reinit(cell, face_index);
      }
}

int
main()
{
  test<3>();
}
