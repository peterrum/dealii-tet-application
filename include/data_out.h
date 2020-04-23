#ifndef TET_DATA_OUT
#define TET_DATA_OUT

using namespace dealii;

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

      const UpdateFlags flag = update_quadrature_points;

      std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

      unsigned int n_cells = 0;

      auto global_to_local = [&](const types::global_dof_index index) {
        if (is_local.is_element(index))
          return is_local.index_within_set(index);

        if (is_ghost.is_element(index))
          return is_ghost.index_within_set(index) + is_local.n_elements();

        Assert(false, ExcNotImplemented());
      };

      // prepare points
      for (const auto &cell : dof_handler.cell_iterators())
        {
          if (!cell->is_locally_owned())
            continue;

          n_cells++;

          FEValues<dim, spacedim> fe_values(mapping, fe, quad, flag);
          fe_values.reinit(cell);

          cell->get_dof_indices(dof_indices);

          const auto &points = fe_values.get_quadrature_points();

          for (unsigned int i = 0; i < dofs_per_cell; i++)
            all_points[global_to_local(dof_indices[i])] = points[i];
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

          FEValues<dim, spacedim> fe_values(mapping, fe, quad, flag);
          fe_values.reinit(cell);

          cell->get_dof_indices(dof_indices);

          stream << dofs_per_cell << " ";

          for (unsigned int i = 0; i < dofs_per_cell; i++)
            stream << global_to_local(dof_indices[i]) << " ";

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

      for (unsigned int i = 0; i < n_dofs; ++i)
        stream << vector.local_element(i) << std::endl;
    }
  } // namespace Tet
} // namespace dealii

#endif