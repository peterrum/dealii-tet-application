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

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

using namespace dealii;

template <int dim, int spacedim = dim>
void
test(const std::string &file_name)
{
  Triangulation<dim, spacedim> tria;

  GridIn<dim, spacedim> grid_in;
  grid_in.attach_triangulation(tria);
  std::ifstream input_file(file_name);

  if (false)
    grid_in.read_ucd(input_file);
  else if (false)
    grid_in.read_msh(input_file);
  else if (false)
    grid_in.read_abaqus(input_file);
  else if (true)
    grid_in.read_unv(input_file);

  // 2) Output generated triangulation via GridOut
  GridOut       grid_out;
  std::ofstream out("mesh.out.vtk");
  grid_out.write_vtk(tria, out);
}

int
main(int argc, char **argv)
{
  if (argc < 3)
    exit(1);

  int dim = atoi(argv[1]);

  if (dim == 2)
    test<2>(argv[2]);
  else if (dim == 3)
    test<3>(argv[2]);
}
