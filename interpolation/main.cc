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
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/tet/fe_q.h>
#include <deal.II/tet/grid_generator.h>
#include <deal.II/tet/mapping_q.h>
#include <deal.II/tet/quadrature_lib.h>

#include "../include/data_out.h"

using namespace dealii;

template <int dim>
struct Parameters
{
  unsigned int degree = 2;


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

template <int dim, int spacedim>
MPI_Comm
get_communicator(const Triangulation<dim, spacedim> &tria)
{
  if (auto tria_ =
        dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(&tria))
    return tria_->get_communicator();

  if (auto tria_ =
        dynamic_cast<const Tet::Triangulation<dim, spacedim> *>(&tria))
    return tria_->get_communicator();

  return MPI_COMM_SELF;
}

template <int dim>
class LinearFunction : public Function<dim>
{
public:
  LinearFunction()
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const
  {
    (void)component;

    // return p[0] <=1.05 ? p[0] : (p[0]-1.1);
    return p[1];
  }
};

template <int dim, int spacedim = dim>
void
test(const Triangulation<dim, spacedim> &tria,
     const FiniteElement<dim, spacedim> &fe,
     const Mapping<dim, spacedim> &      mapping)
{
  DoFHandler<dim, spacedim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  const MPI_Comm comm = get_communicator(dof_handler.get_triangulation());

  IndexSet locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

  using VectorType = LinearAlgebra::distributed::Vector<double>;

  VectorType solution;

  solution.reinit(dof_handler.locally_owned_dofs(),
                  locally_relevant_dofs,
                  comm);

  // VectorTools::interpolate(mapping, dof_handler,  ConstantFunction<dim>(2),
  // solution);
  VectorTools::interpolate(mapping,
                           dof_handler,
                           LinearFunction<dim>(),
                           solution);

  if (dynamic_cast<const Tet::Triangulation<dim, spacedim> *>(&tria) == nullptr)
    {
      solution.update_ghost_values();

      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;

      DataOut<dim> data_out;
      data_out.set_flags(flags);
      data_out.attach_dof_handler(dof_handler);
      data_out.add_data_vector(solution, "solution");
      // data_out.build_patches();
      data_out.build_patches(mapping, fe.degree);
      std::ofstream output(
        "solution." + std::to_string(Utilities::MPI::this_mpi_process(comm)) +
        ".vtk");
      data_out.write_vtk(output);
    }
  else
    {
      solution.update_ghost_values();

      std::ofstream output(
        "solution_tet." +
        std::to_string(Utilities::MPI::this_mpi_process(comm)) + ".vtk");
      Tet::data_out(dof_handler, solution, "solution", output);
    }
}

template <int dim, int spacedim>
void
partition_triangulation(const unsigned                     n_part,
                        Tet::Triangulation<dim, spacedim> &tria)
{
  // determine number of cells (TODO)
  const unsigned int n_cells = std::distance(tria.begin(), tria.end());

  // determine number of cells per process
  const unsigned int n_cells_per_proc = (n_cells + n_part - 1) / n_part;

  // partition mesh
  unsigned int counter = 0;
  for (const auto &cell : tria.cell_iterators())
    cell->set_subdomain_id(counter++ / n_cells_per_proc);

  // collect vertices of locally owned cells
  std::vector<bool> vertex_of_own_cell(tria.n_vertices(), false);
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned())
      for (unsigned int v = 0; v < 3 /*TODO*/; v++)
        vertex_of_own_cell[cell->vertex_index(v)] = true;

  // clear artificial cells
  for (const auto &cell : tria.cell_iterators())
    {
      const auto temp = cell->subdomain_id();
      cell->set_subdomain_id(numbers::artificial_subdomain_id);

      for (unsigned int v = 0; v < 3 /*TODO*/; v++)
        if (vertex_of_own_cell[cell->vertex_index(v)])
          {
            cell->set_subdomain_id(temp);
            break;
          }
    }
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

  // ... partition it
  partition_triangulation(Utilities::MPI::n_mpi_processes(comm), tria);

  // 2) Output generated triangulation via GridOut
  GridOut       grid_out;
  std::ofstream out(params.file_name_out + "." +
                    std::to_string(Utilities::MPI::this_mpi_process(comm)) +
                    ".vtk");
  grid_out.write_vtk(tria, out);

  // 3) Select components
  Tet::FE_Q<dim> fe(params.degree);

  Tet::MappingQ<dim> mapping(1);

  // 4) Perform test (independent of mesh type)
  test(tria, fe, mapping);
}

template <int dim, int spacedim = dim>
void
test_hex(const MPI_Comm &comm, const Parameters<dim> &params)
{
  // 1) Create triangulation...
  parallel::distributed::Triangulation<dim, spacedim> tria(comm);

  if (params.use_grid_generator)
    {
      // ...via GridGenerator
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

  // 2) Output generated triangulation via GridOut
  GridOut       grid_out;
  std::ofstream out(params.file_name_out + "." +
                    std::to_string(Utilities::MPI::this_mpi_process(comm)) +
                    ".vtk");
  grid_out.write_vtk(tria, out);

  // 3) Select components
  FE_Q<dim> fe(params.degree);

  MappingQ<dim, spacedim> mapping(1);

  // 4) Perform test (independent of mesh type)
  test(tria, fe, mapping);
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  const MPI_Comm comm = MPI_COMM_WORLD;

  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(comm) == 0);

  // 2D
  if constexpr (true)
    {
      // setup parameters: TODO move to json file
      Parameters<2> params;
      params.use_grid_generator = true;
      params.repetitions        = std::vector<unsigned int>{20, 20};

      // test TRI
      {
        pcout << "Solve problem on TRI mesh:" << std::endl;

        params.file_name_out = "mesh-tri";
        params.p1            = Point<2>(0, 0);
        params.p2            = Point<2>(1, 1);
        test_tet(comm, params);
      }

      // test HEX
      {
        pcout << "Solve problem on QUAD mesh:" << std::endl;

        params.file_name_out = "mesh-quad";
        params.p1            = Point<2>(1.1, 0); // shift to the right for
        params.p2            = Point<2>(2.1, 1); // visualization purposes
        test_hex(comm, params);
      }
    }

  // 3D
  if constexpr (false)
    {
      // setup parameters: TODO move to json file
      Parameters<3> params;
      params.use_grid_generator = true;
      params.repetitions        = std::vector<unsigned int>{2, 2, 2};

      // test TET
      {
        pcout << "Solve problem on TET mesh:" << std::endl;

        params.file_name_out = "mesh-tet";
        params.p1            = Point<3>(0, 0, 0);
        params.p2            = Point<3>(1, 1, 1);
        test_tet(comm, params);
      }
    }
}