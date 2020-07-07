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
  ScratchData(const Mapping<dim> &       mapping,
              const FiniteElement<dim> & fe,
              const Quadrature<dim> &    quad,
              const Quadrature<dim - 1> &quad_face,
              const UpdateFlags          update_flags = update_values |
                                               update_gradients |
                                               update_quadrature_points |
                                               update_JxW_values,
              const UpdateFlags interface_update_flags =
                update_values | update_gradients | update_quadrature_points |
                update_JxW_values | update_normal_vectors)
    : fe_values(mapping, fe, quad, update_flags)
    , fe_interface_values(mapping, fe, quad_face, interface_update_flags)
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

  // quadrature rules
  Tet::QGauss<dim> quad(dim == 2 ? (degree == 1 ? 3 : 7) :
                                   (degree == 1 ? 4 : 10));

  Tet::QGauss<dim - 1> quad_face(dim == 2 ? (degree == 1 ? 2 : 3) :
                                            (degree == 1 ? 3 : 7));

  const auto cell_worker =
    [&](const auto &cell, auto &scratch_data, auto &copy_data) {
      const unsigned int n_dofs = scratch_data.fe_values.get_fe().dofs_per_cell;
      copy_data.reinit(cell, n_dofs);
      scratch_data.fe_values.reinit(cell);
      const auto &q_points = scratch_data.fe_values.get_quadrature_points();
      const FEValues<dim> &      fe_v = scratch_data.fe_values;
      const std::vector<double> &JxW  = fe_v.get_JxW_values();

      for (unsigned int q = 0; q < fe_v.n_quadrature_points; ++q)
        for (unsigned int i = 0; i < n_dofs; ++i)
          for (unsigned int j = 0; j < n_dofs; ++j)
            copy_data.cell_matrix(i, j) +=
              *fe_v.fe_values.shape_grad(i, q)  // grad phi_i(x_q)
              * fe_v.fe_values.shape_grad(j, q) // grad phi_j(x_q)
              * JxW[q];                         // dx
    };

  const auto boundary_worker = [&](const auto &cell,
                                   const auto &face_no,
                                   auto &      scratch_data,
                                   auto &      copy_data) {
    (void)cell;
    (void)face_no;
    (void)scratch_data;
    (void)copy_data;

    // TODO
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

    FEInterfaceValues<dim> &fe_iv = scratch_data.fe_interface_values;
    fe_iv.reinit(cell, f, sf, ncell, nf, nsf);
    const auto &q_points = fe_iv.get_quadrature_points();
    copy_data.face_data.emplace_back();
    CopyDataFace &     copy_data_face = copy_data.face_data.back();
    const unsigned int n_dofs         = fe_iv.n_current_interface_dofs();
    copy_data_face.joint_dof_indices  = fe_iv.get_interface_dof_indices();
    copy_data_face.cell_matrix.reinit(n_dofs, n_dofs);
    const std::vector<double> &        JxW     = fe_iv.get_JxW_values();
    const std::vector<Tensor<1, dim>> &normals = fe_iv.get_normal_vectors();

    for (unsigned int qpoint = 0; qpoint < q_points.size(); ++qpoint)
      for (unsigned int i = 0; i < n_dofs; ++i)
        for (unsigned int j = 0; j < n_dofs; ++j)
          copy_data_face.cell_matrix(i, j) +=
            (fe_iv.shape_value(true, i, qpoint)       // phi_i
               * fe_iv.shape_value(true, j, qpoint)   // phi_j
               * 1.0                                  // tau
             + fe_iv.shape_value(true, i, qpoint)     // phi_i
                 * fe_iv.shape_value(true, j, qpoint) // phi_j
                 * 1.0                                // tau
             ) *
            JxW(qpoint); // dx
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
      constraints.distribute_local_to_global(cdf.cell_matrix,
                                             cdf.joint_dof_indices,
                                             system_matrix);
  };

  ScratchData<dim> scratch_data(mapping, fe, quad, quad_face);

  CopyData copy_data;
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
