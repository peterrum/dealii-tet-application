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

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_isoparametric.h>

#include <deal.II/grid/tria.h>

#include <deal.II/tet/polynomials.h>
#include <deal.II/tet/quadrature_lib.h>

using namespace dealii;

template <int dim>
class PProjector
{
public:
  static Quadrature<3>
  project_to_all_faces(const Quadrature<2> quad)
  {
    Assert(false, ExcNotImplemented());

    return Quadrature<3>();
  }

  static Quadrature<2>
  project_to_all_faces(const Quadrature<1> quad)
  {
    const auto &sub_quadrature_points  = quad.get_points();
    const auto &sub_quadrature_weights = quad.get_weights();

    const std::array<std::pair<std::array<Point<2>, 2>, unsigned int>, 3>
      faces = {{{{Point<2>(0.0, 0.0), Point<2>(1.0, 0.0)}, 1.0},
                {{Point<2>(1.0, 0.0), Point<2>(0.0, 1.0)}, 1.41421356237},
                {{Point<2>(0.0, 1.0), Point<2>(0.0, 0.0)}, 1.0}}};

    Tet::ScalarPolynomial<1> poly(1);

    std::vector<Point<2>> points;
    std::vector<double>   weights;

    for (const auto &face : faces)
      {
        for (unsigned int o = 0; o < 2; ++o)
          {
            std::array<Point<2>, 2> support_points;

            switch (o)
              {
                case 0:
                  support_points = {face.first[1], face.first[0]};
                  break;
                case 1:
                  support_points = {face.first[0], face.first[1]};
                  break;
                default:
                  Assert(false, ExcNotImplemented());
              }

            for (unsigned int j = 0; j < sub_quadrature_points.size(); ++j)
              {
                Point<2> mapped_point;

                for (unsigned int i = 0; i < 2; ++i)
                  mapped_point +=
                    support_points[i] *
                    poly.compute_value(i, sub_quadrature_points[j]);

                points.push_back(mapped_point);
                weights.push_back(sub_quadrature_weights[j] * face.second);

                std::cout << mapped_point << std::endl;
              }
          }
        std::cout << std::endl;
      }

    return {points, weights};
  }
};

template <int dim, int spacedim = dim>
void
test()
{
  const unsigned int degree = 2;

  // quadrature rule
  Tet::QGauss<dim - 1> quad(dim == 2 ? (degree == 1 ? 2 : 3) :
                                       (degree == 1 ? 3 : 7));

  PProjector<dim>::project_to_all_faces(quad);
}

int
main()
{
  test<2>();
}
