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


#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/numerics/data_out.h>

#include <deal.II/tet/data_out.h>
#include <deal.II/tet/fe_q.h>
#include <deal.II/tet/grid_generator.h>
#include <deal.II/tet/mapping_q.h>
#include <deal.II/tet/partition.h>
#include <deal.II/tet/quadrature_lib.h>

using namespace dealii;

template <int dim, int spacedim = dim>
void
test(const std::string &file_name)
{
  Triangulation<dim, spacedim> tria;

  GridIn<dim, spacedim> grid_in;
  grid_in.attach_triangulation(tria);
  std::ifstream input_file(file_name);
  // grid_in.read_ucd(input_file);
  // grid_in.read_msh(input_file);
  // grid_in.read_abaqus(input_file);
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
