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

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/numerics/data_out.h>

#include <deal.II/tet/fe_q.h>
#include <deal.II/tet/grid_generator.h>
#include <deal.II/tet/mapping_q.h>
#include <deal.II/tet/quadrature_lib.h>

#include "../include/data_out.h"
#include "../include/partition.h"

using namespace dealii;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  const MPI_Comm comm = MPI_COMM_WORLD;

  if (argc == 3) // 2D
    {
      Tet::Triangulation<2, 2> tria(comm, true);

      std::vector<unsigned int> reps{atoi(argv[1]), atoi(argv[2])};
      Point<2>                  p1(0, 0);
      Point<2>                  p2(1, 1);

      Triangulation<2, 2> tria_hex;
      GridGenerator::subdivided_hyper_rectangle(tria_hex, reps, p1, p2, false);

      Tet::GridGenerator::hex_to_tet_grid(tria_hex, tria);

      GridOut       grid_out;
      std::ofstream out("grid.vtk");
      grid_out.write_vtk(tria, out);
    }
  else if (argc == 4) // 3D
    {
      Tet::Triangulation<3, 3> tria(comm, true);

      // clang-format off
      std::vector<unsigned int> reps{atoi(argv[1]), atoi(argv[2]), atoi(argv[3])};
      Point<3>                  p1(0, 0, 0);
      Point<3>                  p2(1, 1, 1);
      // clang-format on

      Triangulation<3, 3> tria_hex;
      GridGenerator::subdivided_hyper_rectangle(tria_hex, reps, p1, p2, false);

      Tet::GridGenerator::hex_to_tet_grid(tria_hex, tria);

      GridOut       grid_out;
      std::ofstream out("grid.vtk");
      grid_out.write_vtk(tria, out);
    }
  else
    {
      AssertThrow(false, ExcNotImplemented());
    }
}
