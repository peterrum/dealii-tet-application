/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2016 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 */

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_isoparametric.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/meshworker/mesh_loop.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/simplex/data_out.h>
#include <deal.II/simplex/fe_lib.h>
#include <deal.II/simplex/grid_generator.h>
#include <deal.II/simplex/quadrature_lib.h>

#include <fstream>
#include <iostream>

using namespace dealii;

void
test(const unsigned int degree)
{
  const int dim      = 2;
  const int spacedim = 2;

  Triangulation<dim, spacedim> tria;

  std::vector<Point<spacedim>> points;
  points.emplace_back(0, 0);
  points.emplace_back(1, 0);
  points.emplace_back(0, 1);
  points.emplace_back(1, 1);
  points.emplace_back(2, 1);
  points.emplace_back(1, 2);

  std::vector<CellData<dim>> cells;

  {
    CellData<dim> cell;
    cell.vertices = {3, 4, 5};
    cells.emplace_back(cell);
  }
  {
    CellData<dim> cell;
    cell.vertices = {0, 1, 2, 3};
    cells.emplace_back(cell);
  }
  {
    CellData<dim> cell;
    cell.vertices = {1, 4, 3};
    cells.emplace_back(cell);
  }
  {
    CellData<dim> cell;
    cell.vertices = {2, 3, 5};
    cells.emplace_back(cell);
  }

  tria.create_triangulation(points, cells, SubCellData());

  DoFHandler<dim, spacedim> dof_handler(tria, /*hp=*/true);

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->reference_cell_type() == ReferenceCell::Type::Tri)
      cell->set_active_fe_index(0);
    else
      cell->set_active_fe_index(1);

  Simplex::FE_P<dim, spacedim>    fe1(degree);
  FE_Q<dim, spacedim>             fe2(degree);
  hp::FECollection<dim, spacedim> fes(fe1, fe2);

  dof_handler.distribute_dofs(fes);

  std::vector<types::global_dof_index> dof_indices;
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      dof_indices.resize(fes[cell->active_fe_index()].n_dofs_per_cell());

      cell->get_dof_indices(dof_indices);

      for (const auto &i : dof_indices)
        std::cout << i << " ";
      std::cout << std::endl;
    }

  GridOut       grid_out;
  std::ofstream out("hp-mesh.vtk");
  grid_out.write_vtk(tria, out);
}

int
main()
{
  deallog.depth_console(0);

  test(/*degree=*/1);
  std::cout << std::endl << std::endl;

  test(/*degree=*/2);

  return 0;
}