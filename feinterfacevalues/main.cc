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

#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_isoparametric.h>
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

#include <deal.II/meshworker/mesh_loop.h>

#include <deal.II/numerics/data_out.h>

#include <deal.II/tet/data_out.h>
#include <deal.II/tet/fe_q.h>
#include <deal.II/tet/grid_generator.h>
#include <deal.II/tet/mapping_q.h>
#include <deal.II/tet/partition.h>
#include <deal.II/tet/quadrature_lib.h>

using namespace dealii;

template <int dim>
struct ScratchData
{
  ScratchData(const Mapping<dim> &      mapping,
              const FiniteElement<dim> &fe,
              const unsigned int        quadrature_degree,
              const UpdateFlags         update_flags = update_values |
                                               update_gradients |
                                               update_quadrature_points |
                                               update_JxW_values,
              const UpdateFlags interface_update_flags =
                update_values | update_gradients | update_quadrature_points |
                update_JxW_values | update_normal_vectors)
    : fe_values(mapping, fe, QGauss<dim>(quadrature_degree), update_flags)
    , fe_interface_values(mapping,
                          fe,
                          QGauss<dim - 1>(quadrature_degree),
                          interface_update_flags)
  {}
  ScratchData(const ScratchData<dim> &scratch_data)
    : fe_values(scratch_data.fe_values.get_mapping(),
                scratch_data.fe_values.get_fe(),
                scratch_data.fe_values.get_quadrature(),
                scratch_data.fe_values.get_update_flags())
    , fe_interface_values(
        scratch_data.fe_values
          .get_mapping(), // TODO: implement for fe_interface_values
        scratch_data.fe_values.get_fe(),
        scratch_data.fe_interface_values.get_quadrature(),
        scratch_data.fe_interface_values.get_update_flags())
  {}
  FEValues<dim>          fe_values;
  FEInterfaceValues<dim> fe_interface_values;
};



struct CopyDataFace
{
  FullMatrix<double>                   cell_matrix;
  std::vector<types::global_dof_index> joint_dof_indices;
};



struct CopyData
{
  FullMatrix<double>                   cell_matrix;
  Vector<double>                       cell_rhs;
  std::vector<types::global_dof_index> local_dof_indices;
  std::vector<CopyDataFace>            face_data;
  template <class Iterator>
  void
  reinit(const Iterator &cell, unsigned int dofs_per_cell)
  {
    cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
    cell_rhs.reinit(dofs_per_cell);
    local_dof_indices.resize(dofs_per_cell);
    cell->get_dof_indices(local_dof_indices);
  }
};


template <int dim, int spacedim = dim>
void
test(const unsigned int degree)
{
  // create triangulation
  Triangulation<dim, spacedim> tria;
  std::vector<unsigned int>    sub{2, 2};
  Point<dim>                   p1(0, 0);
  Point<dim>                   p2(1, 1);
  Tet::GridGenerator::subdivided_hyper_rectangle(tria, sub, p1, p2, false);

  // mapping
  Tet::FE_Q<dim>            fe_mapping(1);
  MappingIsoparametric<dim> mapping(fe_mapping);

  // finite element
  Tet::FE_Q<dim> fe(degree);

  DoFHandler<dim, spacedim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  // quadrature rule
  Tet::QGauss<dim - 1> quad(dim == 2 ? (degree == 1 ? 2 : 3) :
                                       (degree == 1 ? 3 : 7));

  const auto cell_worker =
    [&](const auto &cell, auto &scratch_data, auto &copy_data) {
      (void)cell;
      (void)scratch_data;
      (void)copy_data;
    };

  const auto boundary_worker = [&](const auto &cell,
                                   const auto &face_no,
                                   auto &      scratch_data,
                                   auto &      copy_data) {
    (void)cell;
    (void)face_no;
    (void)scratch_data;
    (void)copy_data;
  };

  const auto face_worker = [&](const auto &cell,
                               const auto &f,
                               const auto &sf,
                               const auto &ncell,
                               const auto &nf,
                               const auto &nsf,
                               auto &      scratch_data,
                               auto &      copy_data) {
    (void)cell;
    (void)f;
    (void)sf;
    (void)ncell;
    (void)nf;
    (void)nsf;
    (void)scratch_data;
    (void)copy_data;
  };

  SparseMatrix<double> system_matrix;
  Vector<double>       right_hand_side;

  const AffineConstraints<double> constraints;

  const auto copier = [&](const auto &c) {
    constraints.distribute_local_to_global(c.cell_matrix,
                                           c.cell_rhs,
                                           c.local_dof_indices,
                                           system_matrix,
                                           right_hand_side);
    for (auto &cdf : c.face_data)
      {
        constraints.distribute_local_to_global(cdf.cell_matrix,
                                               cdf.joint_dof_indices,
                                               system_matrix);
      }
  };

  const unsigned int n_gauss_points = degree + 1;
  ScratchData<dim>   scratch_data(mapping, fe, n_gauss_points);
  CopyData           copy_data;
  MeshWorker::mesh_loop(dof_handler.begin_active(),
                        dof_handler.end(),
                        cell_worker,
                        copier,
                        scratch_data,
                        copy_data,
                        MeshWorker::assemble_own_cells |
                          MeshWorker::assemble_boundary_faces |
                          MeshWorker::assemble_own_interior_faces_once,
                        boundary_worker,
                        face_worker);
}

int
main()
{
  test<2>(1 /*=fe_degree*/);
}
