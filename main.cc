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


#include <deal.II/base/mpi.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/tet/fe_q.h>
#include <deal.II/tet/grid_generator.h>

using namespace dealii;

template <int dim>
struct Parameters
{
  unsigned int degree = 1;


  // GridGenerator
  bool                      use_grid_generator = true;
  std::vector<unsigned int> repetitions;
  Point<dim>                p1;
  Point<dim>                p2;

  // GridIn
  std::string file_name_in = "";

  // GridOut
  std::string file_name_out = "";
};

template <int dim, int spacedim = dim>
void
test(const Triangulation<dim, spacedim> &tria,
     const FiniteElement<dim, spacedim> &fe)
{
  DoFHandler<dim, spacedim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);
}

template <int dim, int spacedim = dim>
void
test_tet(const MPI_Comm &comm, const Parameters<dim> &params)
{
  // 1) Create triangulation...
  Tet::Triangulation<dim, spacedim> tria(comm);

  if (params.use_grid_generator)
    {
      // ...via Tet::GridGenerator
      Tet::GridGenerator::subdivided_hyper_rectangle(
        tria, params.repetitions, params.p1, params.p2, false);
    }
  else
    {
      // ...via GridIn
      GridIn<dim, spacedim> grid_in;
      grid_in.attach_triangulation(tria);
      std::ifstream input_file(params.file_name_in);
      grid_in.read_ucd(input_file);
    }

  // 2) Output generated triangulation via
  GridOut       grid_out;
  std::ofstream out(params.file_name_out + "." +
                    std::to_string(Utilities::MPI::this_mpi_process(comm)) +
                    ".vtk");
  grid_out.write_vtk(tria, out);

  // 3) Select components
  Tet::FE_Q<dim> fe(params.degree);

  // 4) Perform test (independent of mesh type)
  test(tria, fe);
}

template <int dim, int spacedim = dim>
void
test_quad(const MPI_Comm &comm, const Parameters<dim> &params)
{
  // 1) Create triangulation...
  parallel::distributed::Triangulation<dim, spacedim> tria(comm);

  if (params.use_grid_generator)
    {
      // ...via Tet::GridGenerator
      GridGenerator::subdivided_hyper_rectangle(
        tria, params.repetitions, params.p1, params.p2, false);
    }
  else
    {
      // ...via GridIn
      GridIn<dim, spacedim> grid_in;
      grid_in.attach_triangulation(tria);
      std::ifstream input_file(params.file_name_in);
      grid_in.read_ucd(input_file);
    }

  // 2) Output generated triangulation via
  GridOut       grid_out;
  std::ofstream out(params.file_name_out + "." +
                    std::to_string(Utilities::MPI::this_mpi_process(comm)) +
                    ".vtk");
  grid_out.write_vtk(tria, out);

  // 3) Select components
  FE_Q<dim> fe(params.degree);

  // 4) Perform test (independent of mesh type)
  test(tria, fe);
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  // setup parameters: TODO move to json file
  Parameters<2> params;
  params.use_grid_generator = true;
  params.repetitions        = std::vector<unsigned int>{3, 3};

  const MPI_Comm comm = MPI_COMM_WORLD;

  // test TET
  {
    params.file_name_out = "mesh-tet";
    params.p1            = Point<2>(0, 0);
    params.p2            = Point<2>(1, 1);
    test_tet(comm, params);
  }

  // test QUAD
  {
    params.file_name_out = "mesh-quad";
    params.p1            = Point<2>(1.1, 0); // shift to the right for
    params.p2            = Point<2>(2.1, 1); // visualization purposes
    test_quad(comm, params);
  }
}