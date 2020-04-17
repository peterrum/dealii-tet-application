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

const MPI_Comm comm = MPI_COMM_WORLD;

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
test_tet(const Parameters<dim> &params)
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
  std::ofstream out(params.file_name_out);
  grid_out.write_vtk(tria, out);

  // 3) Select components
  Tet::FE_Q<dim> fe(params.degree);

  // 4) Perform test (independent of mesh type)
  test(tria, fe);
}

template <int dim, int spacedim = dim>
void
test_quad(const Parameters<dim> &params)
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
  std::ofstream out(params.file_name_out);
  grid_out.write_vtk(tria, out);

  // 3) Select components
  FE_Q<dim> fe(params.degree);

  // 4) Perform test (independent of mesh type)
  test(tria, fe);
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize(argc, argv, 1);

  Parameters<2> params;
  params.use_grid_generator = true;
  params.repetitions        = std::vector<unsigned int>{1, 1};
  params.p1                 = Point<2>(0, 0);
  params.p2                 = Point<2>(1, 1);

  // test TET
  {
    std::cout << "A" << std::endl;
    params.file_name_out = "mesh-tet.%d.vtk";
    test_tet(params);
  }

  // test QUAD
  {
    std::cout << "B" << std::endl;
    params.file_name_out = "mesh-quad.%d.vtk";
    test_tet(params);
  }
}