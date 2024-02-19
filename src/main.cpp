#include "../include/include.h"
#include "../include/parameters.h"
#include "../include/utility.h"
#include "../include/marching_tetrahedrons.h"
#include "../include/viz_mesh.h"
#include "../include/save_ply.h"

int main(int argc, char* argv[])
{
    // ===============================================================
    // Generate Pointcloud with Random density
    auto start_gen_pointcloud = std::chrono::high_resolution_clock::now();

    std::vector<cv::Point3f> pointcloud = generate_random_grid();
    PointCloud pointcloud_with_density;
    if(READ_FILE)
    {
        cv::String ply_path = argv[1];
        pointcloud = get_pointcloud_from_txt(ply_path);
    }
    pointcloud_with_density = add_random_density(pointcloud);
    std::cout << "Number of pointcloud: " << pointcloud.size() << std::endl;

    auto end_gen_pointcloud = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gen_pointcloud_duration = end_gen_pointcloud - start_gen_pointcloud;
    std::cout << "Pointcloud Generation Time: " << gen_pointcloud_duration.count() << " ms" << std::endl;
    // ===============================================================

    // ===============================================================
    // Calculate Voxel Size
    auto start_cal_voxel_size = std::chrono::high_resolution_clock::now();

    float min_x, min_y, min_z;
    find_min_pixel(pointcloud, min_x, min_y, min_z);
    float max_x, max_y, max_z;
    find_max_pixel(pointcloud, max_x, max_y, max_z);
    
    float voxel_dx = 1;
    float voxel_dy = 1;
    float voxel_dz = 1;
    if(READ_FILE)
        cal_voxel_size(min_x, min_y, min_z, max_x, max_y, max_z, voxel_dx, voxel_dy, voxel_dz);

    auto end_cal_voxel_size = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cal_voxel_size_duration = end_cal_voxel_size - start_cal_voxel_size;
    std::cout << "Voxel Size Calculation Time: " << cal_voxel_size_duration.count() << " ms" << std::endl;
    // ===============================================================

    // ===============================================================
    // Marching Cubes
    auto start_marching_cubes = std::chrono::high_resolution_clock::now();

    std::vector<Triangle> triangles;
    for (float z = min_z - voxel_dz; z <= max_z; z += voxel_dz)
    {
        for (float y = min_y - voxel_dy; y <= max_y; y += voxel_dy)
        {
            for (float x = min_x - voxel_dx; x <= max_x; x += voxel_dx)
            {
                Voxel cur_voxel;

                // Initialize Voxel Info
                init_voxel_vertices(pointcloud_with_density, cur_voxel, x, y, z, voxel_dx, voxel_dy, voxel_dz);

                // Calculate six triangles 
                std::vector<Tetrahedron> cur_six_tetrahedrons;
                divide_into_six_triangles(cur_voxel, cur_six_tetrahedrons);

                // Get six edges rule
                std::vector<std::array<int, 6>> cur_six_edges_rule;
                get_vertice_density(cur_six_tetrahedrons, cur_six_edges_rule);

                // Make triangles
                make_triangle(triangles, cur_six_tetrahedrons, cur_six_edges_rule);
            }
        }
    }

    auto end_marching_cubes = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> marching_cubes_duration = end_marching_cubes - start_marching_cubes;
    std::cout << "Marching Tetrahedrons Time: " << marching_cubes_duration.count() << " ms" << std::endl;
    // ===============================================================

    // ===============================================================
    // Write PLY file using Triangles
    std::cout << "Number of triangles: " << triangles.size() << std::endl;
    cv::String save_path = argv[2];
    write_to_ply(pointcloud, triangles, save_path.c_str());
    // ===============================================================
    
    return 0;
}