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

#ifndef dealii_simplex_grid_generator_h_
#define dealii_simplex_grid_generator_h_


#include <deal.II/base/config.h>

#include <deal.II/base/point.h>

#include <deal.II/grid/tria.h>

#include <vector>

DEAL_II_NAMESPACE_OPEN

namespace Simplex
{
  /**
   * This namespace provides a collection of functions to generate simplex
   * triangulations for some basic geometries.
   */
  namespace GridGenerator
  {
    /**
     * Create a coordinate-parallel brick from the two diagonally opposite
     * corner points @p p1 and @p p2. The number of vertices in coordinate
     * direction @p i is given by <tt>repetitions[i]+1</tt>.
     *
     * @note This function connects internally 4/8 vertices to quadrilateral/
     *   hexahedral cells and subdivides these into 2/5 triangular/
     *   tetrahedral cells.
     *
     * @note Currently, this function only works for `dim==spacedim`.
     */
    template <int dim, int spacedim>
    void
    subdivided_hyper_rectangle_(Triangulation<dim, spacedim> &   tria,
                                const std::vector<unsigned int> &repetitions,
                                const Point<dim> &               p1,
                                const Point<dim> &               p2,
                                const bool colorize = false)
    {
      AssertDimension(dim, spacedim);

      AssertThrow(colorize == false, ExcNotImplemented());

      std::vector<Point<spacedim>> vertices;
      std::vector<CellData<dim>>   cells;

      if (dim == 2)
        {
          // determine cell sizes
          const Point<dim> dx((p2[0] - p1[0]) / repetitions[0],
                              (p2[1] - p1[1]) / repetitions[1]);

          // create vertices
          for (unsigned int j = 0; j <= repetitions[1]; ++j)
            for (unsigned int i = 0; i <= repetitions[0]; ++i)
              vertices.push_back(
                Point<spacedim>(p1[0] + dx[0] * i, p1[1] + dx[1] * j));

          // create cells
          for (unsigned int j = 0; j < repetitions[1]; ++j)
            for (unsigned int i = 0; i < repetitions[0]; ++i)
              {
                // create reference QUAD cell
                std::array<unsigned int, 4> quad{
                  (j + 0) * (repetitions[0] + 1) + i + 0, //
                  (j + 0) * (repetitions[0] + 1) + i + 1, //
                  (j + 1) * (repetitions[0] + 1) + i + 0, //
                  (j + 1) * (repetitions[0] + 1) + i + 1  //
                };                                        //

                if (j < repetitions[1] / 2 && i < repetitions[0] / 2)
                  {
                    CellData<dim> quad_;
                    quad_.vertices = {quad[0], quad[1], quad[2], quad[3]};
                    cells.push_back(quad_);

                    continue;
                  }

                // TRI cell 0
                {
                  CellData<dim> tri;
                  tri.vertices = {quad[0], quad[1], quad[2]};
                  cells.push_back(tri);
                }

                // TRI cell 1
                {
                  CellData<dim> tri;
                  tri.vertices = {quad[3], quad[2], quad[1]};
                  cells.push_back(tri);
                }
              }
        }
      else
        {
          AssertThrow(colorize == false, ExcNotImplemented());
        }

      // actually create triangulation
      tria.create_triangulation(vertices, cells, SubCellData());
    }
  } // namespace GridGenerator
} // namespace Simplex



DEAL_II_NAMESPACE_CLOSE

#endif
