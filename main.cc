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

namespace dealii
{
  namespace Tet
  {
    template <int dim>
    class QGaussHelper : public Quadrature<dim>
    {
    public:
      QGaussHelper(const unsigned int n_points)
      {
        this->weights.resize(n_points);

        if (dim == 2)
          {
            if (n_points == 3) // DRT::Element::tri3 (TODO: change order)
              {
                this->quadrature_points.emplace_back(1.0, 0.0);
                this->quadrature_points.emplace_back(0.0, 1.0);
                this->quadrature_points.emplace_back(0.0, 0.0);
              }
            else if (n_points == 6) // DRT::Element::tri6
              {
                this->quadrature_points.emplace_back(1.0, 0.0);
                this->quadrature_points.emplace_back(0.0, 1.0);
                this->quadrature_points.emplace_back(0.0, 0.0);
                this->quadrature_points.emplace_back(0.5, 0.5);
                this->quadrature_points.emplace_back(0.0, 0.5);
                this->quadrature_points.emplace_back(0.5, 0.0);
              }
            else
              {
                Assert(false, ExcNotImplemented());
              }
          }
        else if (dim == 3)
          {
            Assert(false, ExcNotImplemented());
          }
        else
          {
            Assert(false, ExcNotImplemented());
          }

        AssertDimension(this->quadrature_points.size(), this->weights.size());
        Assert(this->quadrature_points.size() > 0,
               ExcMessage("No valid quadrature points!"));
      }
    };

    template <int dim, int spacedim, typename VectorType, typename StreamType>
    void
    data_out(const DoFHandler<dim, spacedim> &dof_handler,
             const VectorType &               vector,
             const std::string &              label,
             StreamType &                     stream)
    {
      const auto &is_local = vector.get_partitioner()->locally_owned_range();
      const auto &is_ghost = vector.get_partitioner()->ghost_indices();

      const unsigned int n_dofs = is_local.n_elements() + is_ghost.n_elements();

      const auto &       fe            = dof_handler.get_fe();
      const unsigned int dofs_per_cell = fe.dofs_per_cell;

      const QGaussHelper<dim> quad(dofs_per_cell);
      const MappingQ<dim>     mapping(1);

      std::vector<Point<spacedim>> all_points(n_dofs);

      const UpdateFlags       flag = update_quadrature_points;
      FEValues<dim, spacedim> fe_values(mapping, fe, quad, flag);

      std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

      unsigned int n_cells = 0;

      // prepare points
      for (const auto &cell : dof_handler.cell_iterators())
        {
          if (!cell->is_locally_owned())
            continue;

          n_cells++;

          fe_values.reinit(cell);

          cell->get_dof_indices(dof_indices);

          const auto &points = fe_values.get_quadrature_points();

          for (unsigned int i = 0; i < dofs_per_cell; i++)
            all_points[dof_indices[i]] = points[i];
        }

      stream << "# vtk DataFile Version 2.0" << std::endl;
      stream << "Cube example" << std::endl;
      stream << "ASCII" << std::endl;
      stream << "DATASET UNSTRUCTURED_GRID" << std::endl;

      stream << "POINTS " << all_points.size() << " float" << std::endl;
      for (const auto &point : all_points)
        {
          for (int d = 0; d < spacedim; ++d)
            stream << point[d] << " ";
          for (int d = spacedim; d < 3; ++d)
            stream << 0.0 << " ";
          stream << std::endl;
        }

      stream << "CELLS " << n_cells << " " << n_cells * (dofs_per_cell + 1)
             << std::endl;

      for (const auto &cell : dof_handler.cell_iterators())
        {
          if (!cell->is_locally_owned())
            continue;

          fe_values.reinit(cell);

          cell->get_dof_indices(dof_indices);

          stream << dofs_per_cell << " ";

          for (unsigned int i = 0; i < dofs_per_cell; i++)
            stream << dof_indices[i] << " ";

          stream << std::endl;
        }

      stream << "CELL_TYPES " << n_cells << std::endl;

      auto cell_type = [](auto dofs_per_cell) {
        if (dim == 2 && dofs_per_cell == 3)
          return 5;
        if (dim == 2 && dofs_per_cell == 6)
          return 22;

        Assert(false, ExcNotImplemented());
      }(dofs_per_cell);

      for (unsigned int cell = 0; cell < n_cells; cell++)
        stream << cell_type << std::endl;

      stream << "POINT_DATA " << n_dofs << std::endl;
      stream << "SCALARS " << label << " double 1" << std::endl;
      stream << "LOOKUP_TABLE default" << std::endl;

      for (auto i : vector)
        stream << i << std::endl;
    }
  } // namespace Tet
} // namespace dealii

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

template <int dim, int spacedim = dim>
void
test(const Triangulation<dim, spacedim> &tria,
     const FiniteElement<dim, spacedim> &fe,
     const Quadrature<dim> &             quad,
     const Mapping<dim, spacedim> &      mapping)
{
  DoFHandler<dim, spacedim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  AffineConstraints constraint_matrix;
  DoFTools::make_zero_boundary_constraints(dof_handler, constraint_matrix);
  constraint_matrix.close();

  const MPI_Comm comm = get_communicator(dof_handler.get_triangulation());

  IndexSet locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

  using VectorType = LinearAlgebra::distributed::Vector<double>;

  TrilinosWrappers::SparseMatrix system_matrix;
  VectorType                     solution;
  VectorType                     system_rhs;

  TrilinosWrappers::SparsityPattern dsp(dof_handler.locally_owned_dofs(), comm);
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraint_matrix);
  dsp.compress();
  system_matrix.reinit(dsp);

  solution.reinit(dof_handler.locally_owned_dofs(),
                  locally_relevant_dofs,
                  comm);
  system_rhs.reinit(dof_handler.locally_owned_dofs(),
                    locally_relevant_dofs,
                    comm);

  const UpdateFlags flag = update_JxW_values | update_values | update_gradients;
  FEValues<dim, spacedim> fe_values(mapping, fe, quad, flag);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quad.size();

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  for (const auto &cell : dof_handler.cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);
      cell_matrix = 0;
      cell_rhs    = 0;

      for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              cell_matrix(i, j) +=
                (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                 fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                 fe_values.JxW(q_index));           // dx
            cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                            1.0 *                               // 1.0
                            fe_values.JxW(q_index));            // dx
          }

      cell->get_dof_indices(dof_indices);

      constraint_matrix.distribute_local_to_global(
        cell_matrix, cell_rhs, dof_indices, system_matrix, system_rhs);
    }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);

  SolverControl        solver_control(1000, 1e-12);
  SolverCG<VectorType> solver(solver_control);
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());

  std::cout << "   " << solver_control.last_step()
            << " CG iterations needed to obtain convergence." << std::endl;

  system_rhs.print(std::cout);
  solution.print(std::cout);

  if (dynamic_cast<const Tet::Triangulation<dim, spacedim> *>(&tria) == nullptr)
    {
      solution.update_ghost_values();

      DataOut<dim> data_out;
      data_out.attach_dof_handler(dof_handler);
      data_out.add_data_vector(solution, "solution");
      data_out.build_patches();
      std::ofstream output("solution.vtk");
      data_out.write_vtk(output);
    }
  else
    {
      solution.update_ghost_values();

      std::ofstream output("solution_tet.vtk");
      Tet::data_out(dof_handler, solution, "solution", output);
    }

  std::cout << std::endl;
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

  Tet::QGauss<dim> quad(params.degree == 1 ? 3 : 7);

  Tet::MappingQ<dim> mapping(1);

  // 4) Perform test (independent of mesh type)
  test(tria, fe, quad, mapping);
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

  QGauss<dim> quad(params.degree + 1);

  MappingQ<dim, spacedim> mapping(1);

  // 4) Perform test (independent of mesh type)
  test(tria, fe, quad, mapping);
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  // setup parameters: TODO move to json file
  Parameters<2> params;
  params.use_grid_generator = true;
  params.repetitions        = std::vector<unsigned int>{20, 20};

  const MPI_Comm comm = MPI_COMM_WORLD;

  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(comm) == 0);

  // test TET
  {
    pcout << "Solve problem on TET mesh:" << std::endl;

    params.file_name_out = "mesh-tet";
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