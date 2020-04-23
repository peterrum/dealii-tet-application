#ifndef TET_PARTITION
#define TET_PARTITION

using namespace dealii;

namespace dealii
{
  namespace Tet
  {
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


  } // namespace Tet
} // namespace dealii

#endif