/*
    ============================================
    Make Loop Up Table For Marching Tetrahedrons
    ============================================

    Tables and conventions from:
    http://paulbourke.net/geometry/polygonise/

                      + 0
                     /|\
                    / | \
                   /  |  \
                  /   |   \
                 /    |    \
                /     |     \
               +-------------+ 1
              3 \     |     /
                 \    |    /
                  \   |   /
                   \  |  /
                    \ | /
                     \|/
                      + 2


    Vertex : p0, p1, p2, p3
    Edge : a, b, c, d, e, f
    
    Total case : 2^4 = 16
    
    // Not Make Any plane
    {0, 0, 0, 0} <-----> {0, 0, 0, 0, 0, 0} 
    {1, 1, 1, 1} <-----> {0, 0, 0, 0, 0, 0} 

    // Make Triangle
    {1, 0, 0, 0} <-----> {1, 1, 1, 0, 0, 0}
    {0, 1, 1, 1} <-----> {1, 1, 1, 0, 0, 0}
    {0, 1, 0, 0} <-----> {1, 0, 0, 1, 0, 1}
    {1, 0, 1, 1} <-----> {1, 0, 0, 1, 0, 1}
    {0, 0, 1, 0} <-----> {0, 1, 0, 1, 1, 0}
    {1, 1, 0, 1} <-----> {0, 1, 0, 1, 1, 0}
    {0, 0, 0, 1} <-----> {0, 0, 1, 0, 1, 1}
    {1, 1, 1, 0} <-----> {0, 0, 1, 0, 1, 1}

    // Make Square
    {1, 1, 0, 0} <-----> {0, 1, 1, 1, 0, 1}
    {0, 0, 1, 1} <-----> {0, 1, 1, 1, 0, 1}
    {1, 0, 0, 1} <-----> {1, 1, 0, 0, 1, 1}
    {0, 1, 1, 0} <-----> {1, 1, 0, 0, 1, 1}
    {0, 1, 0, 1} <-----> {1, 0, 1, 1, 1, 0}
    {1, 0, 1, 0} <-----> {1, 0, 1, 1, 1, 0}    
*/

#ifndef MARCHING_TETRAHEDRONS
#define MARCHING_TETRAHEDRONS

#include "include.h"
#include "parameters.h"

cv::Point3f interpolation(cv::Point3f pt1, cv::Point3f pt2, 
                          float pt1_density, float pt2_density, float isovalue)
{
    float mu = (isovalue - pt1_density) / (pt2_density - pt1_density);

    float inter_x = pt1.x + mu * (pt2.x - pt1.x);
    float inter_y = pt1.y + mu * (pt2.y - pt1.y);
    float inter_z = pt1.z + mu * (pt2.z - pt1.z);

    return cv::Point3f(inter_x, inter_y, inter_z);
}

void init_voxel_vertices(PointCloud pointcloud, Voxel &voxel, 
                         float cur_x, float cur_y, float cur_z,
                         float diff_x, float diff_y, float diff_z)
{
    cv::Point3f v0(cur_x,          cur_y,          cur_z);
    cv::Point3f v1(cur_x + diff_x, cur_y,          cur_z);
    cv::Point3f v2(cur_x + diff_x, cur_y,          cur_z + diff_z);
    cv::Point3f v3(cur_x,          cur_y,          cur_z + diff_z);
    cv::Point3f v4(cur_x,          cur_y + diff_y, cur_z);
    cv::Point3f v5(cur_x + diff_x, cur_y + diff_y, cur_z);
    cv::Point3f v6(cur_x + diff_x, cur_y + diff_y, cur_z + diff_z);
    cv::Point3f v7(cur_x,          cur_y + diff_y, cur_z + diff_z);

    voxel.vertices.push_back(v0);
    voxel.vertices.push_back(v1);
    voxel.vertices.push_back(v2);
    voxel.vertices.push_back(v3);
    voxel.vertices.push_back(v4);
    voxel.vertices.push_back(v5);
    voxel.vertices.push_back(v6);
    voxel.vertices.push_back(v7);

    for(int i = 0; i < voxel.vertices.size(); i++)
    {
        cv::Point3f vertex = voxel.vertices[i];
        auto it = std::find(pointcloud.vertices.begin(), pointcloud.vertices.end(), vertex);

        if(it == pointcloud.vertices.end())
            voxel.density.push_back(1);            
        else
        {
            int idx = it - pointcloud.vertices.begin();
            voxel.density.push_back(pointcloud.density[idx - 1]);
        }
    }
}

void divide_into_six_triangles(Voxel cur_voxel, std::vector<Tetrahedron> cur_six_tetrahedrons)
{
    // Tetrahedron 1 (v3, v4, v5, v7) => p0 = v3 / p1 = v7 / p2 = v4 / p3 = v5
    // Tetrahedron 2 (v3, v5, v6, v7) => p0 = v3 / p1 = v7 / p2 = v5 / p3 = v6
    // Tetrahedron 3 (v0, v3, v4, v5) => p0 = v3 / p1 = v5 / p2 = v4 / p3 = v0
    // Tetrahedron 4 (v0, v1, v3, v5) => p0 = v5 / p1 = v1 / p2 = v0 / p3 = v3
    // Tetrahedron 5 (v1, v2, v3, v5) => p0 = v5 / p1 = v1 / p2 = v3 / p3 = v2
    // Tetrahedron 6 (v2, v3, v5, v6) => p0 = v3 / p1 = v5 / p2 = v2 / p3 = v6

    Tetrahedron t1;
    t1.vertices.push_back(cur_voxel.vertices[3]);
    t1.density.push_back(cur_voxel.density[3]);
    t1.vertices.push_back(cur_voxel.vertices[7]);
    t1.density.push_back(cur_voxel.density[7]);
    t1.vertices.push_back(cur_voxel.vertices[4]);
    t1.density.push_back(cur_voxel.density[4]);
    t1.vertices.push_back(cur_voxel.vertices[5]);
    t1.density.push_back(cur_voxel.density[5]);
    cur_six_tetrahedrons.push_back(t1);

    Tetrahedron t2;
    t2.vertices.push_back(cur_voxel.vertices[3]);
    t2.density.push_back(cur_voxel.density[3]);
    t2.vertices.push_back(cur_voxel.vertices[7]);
    t2.density.push_back(cur_voxel.density[7]);
    t2.vertices.push_back(cur_voxel.vertices[5]);
    t2.density.push_back(cur_voxel.density[5]);
    t2.vertices.push_back(cur_voxel.vertices[6]);
    t2.density.push_back(cur_voxel.density[6]);
    cur_six_tetrahedrons.push_back(t2);

    Tetrahedron t3;
    t3.vertices.push_back(cur_voxel.vertices[3]);
    t3.density.push_back(cur_voxel.density[3]);
    t3.vertices.push_back(cur_voxel.vertices[5]);
    t3.density.push_back(cur_voxel.density[5]);
    t3.vertices.push_back(cur_voxel.vertices[4]);
    t3.density.push_back(cur_voxel.density[4]);
    t3.vertices.push_back(cur_voxel.vertices[0]);
    t3.density.push_back(cur_voxel.density[0]);
    cur_six_tetrahedrons.push_back(t3);

    Tetrahedron t4;
    t4.vertices.push_back(cur_voxel.vertices[5]);
    t4.density.push_back(cur_voxel.density[5]);
    t4.vertices.push_back(cur_voxel.vertices[1]);
    t4.density.push_back(cur_voxel.density[1]);
    t4.vertices.push_back(cur_voxel.vertices[0]);
    t4.density.push_back(cur_voxel.density[0]);
    t4.vertices.push_back(cur_voxel.vertices[3]);
    t4.density.push_back(cur_voxel.density[3]);
    cur_six_tetrahedrons.push_back(t4);
    
    Tetrahedron t5;
    t5.vertices.push_back(cur_voxel.vertices[5]);
    t5.density.push_back(cur_voxel.density[5]);
    t5.vertices.push_back(cur_voxel.vertices[1]);
    t5.density.push_back(cur_voxel.density[1]);
    t5.vertices.push_back(cur_voxel.vertices[3]);
    t5.density.push_back(cur_voxel.density[3]);
    t5.vertices.push_back(cur_voxel.vertices[2]);
    t5.density.push_back(cur_voxel.density[2]);
    cur_six_tetrahedrons.push_back(t5);

    Tetrahedron t6;
    t6.vertices.push_back(cur_voxel.vertices[3]);
    t6.density.push_back(cur_voxel.density[3]);
    t6.vertices.push_back(cur_voxel.vertices[5]);
    t6.density.push_back(cur_voxel.density[5]);
    t6.vertices.push_back(cur_voxel.vertices[2]);
    t6.density.push_back(cur_voxel.density[2]);
    t6.vertices.push_back(cur_voxel.vertices[6]);
    t6.density.push_back(cur_voxel.density[6]);
    cur_six_tetrahedrons.push_back(t6);
}

void get_vertice_density(std::vector<Tetrahedron> cur_six_tetrahedrons, std::vector<std::array<int, 6>> cur_six_edges_rule)
{
    for(int t = 0; t < cur_six_tetrahedrons.size(); t++)
    {
        Tetrahedron cur_tetrahedron = cur_six_tetrahedrons[t];

        bool p0_value = false;
        bool p1_value = false;
        bool p2_value = false;
        bool p3_value = false;

        if(cur_tetrahedron.density[0] < ISOVALUE)
            p0_value = true;
        if(cur_tetrahedron.density[1] < ISOVALUE)
            p1_value = true;
        if(cur_tetrahedron.density[2] < ISOVALUE)
            p2_value = true;
        if(cur_tetrahedron.density[3] < ISOVALUE)
            p3_value = true;

        std::array<int, 6> vertices_density;
        if(p0_value == false && p1_value == false && p2_value == false && p3_value == false)
            vertices_density = std::array<int, 6>{0, 0, 0, 0, 0, 0};
        else if(p0_value == false && p1_value == false && p2_value == false && p3_value == true)
            vertices_density = std::array<int, 6>{0, 0, 1, 0, 1, 1};
        else if(p0_value == false && p1_value == false && p2_value == true && p3_value == false)
            vertices_density = std::array<int, 6>{0, 1, 0, 1, 1, 0};
        else if(p0_value == false && p1_value == false && p2_value == true && p3_value == true)
            vertices_density = std::array<int, 6>{0, 1, 1, 1, 0, 1};
        else if(p0_value == false && p1_value == true && p2_value == false && p3_value == false)
            vertices_density = std::array<int, 6>{1, 0, 0, 1, 0, 1};
        else if(p0_value == false && p1_value == true && p2_value == false && p3_value == true)
            vertices_density = std::array<int, 6>{1, 0, 1, 1, 1, 0};
        else if(p0_value == false && p1_value == true && p2_value == true && p3_value == false)
            vertices_density = std::array<int, 6>{1, 1, 0, 0, 1, 1};
        else if(p0_value == false && p1_value == true && p2_value == true && p3_value == true)
            vertices_density = std::array<int, 6>{1, 1, 1, 0, 0, 0};
        else if(p0_value == true && p1_value == false && p2_value == false && p3_value == false)
            vertices_density = std::array<int, 6>{1, 1, 1, 0, 0, 0};
        else if(p0_value == true && p1_value == false && p2_value == false && p3_value == true)
            vertices_density = std::array<int, 6>{1, 1, 0, 0, 1, 1};
        else if(p0_value == true && p1_value == false && p2_value == true && p3_value == false)
            vertices_density = std::array<int, 6>{1, 0, 1, 1, 1, 0};
        else if(p0_value == true && p1_value == false && p2_value == true && p3_value == true)
            vertices_density = std::array<int, 6>{1, 0, 0, 1, 0, 1};
        else if(p0_value == true && p1_value == true && p2_value == false && p3_value == false)
            vertices_density = std::array<int, 6>{0, 1, 1, 1, 0, 1};
        else if(p0_value == true && p1_value == true && p2_value == false && p3_value == true)
            vertices_density = std::array<int, 6>{0, 1, 0, 1, 1, 0};
        else if(p0_value == true && p1_value == true && p2_value == true && p3_value == false)
            vertices_density = std::array<int, 6>{0, 0, 1, 0, 1, 1};
        else if(p0_value == true && p1_value == true && p2_value == true && p3_value == true)
            vertices_density = std::array<int, 6>{0, 0, 0, 0, 0, 0};

        cur_six_edges_rule.push_back(vertices_density);
    }
}

void make_triangle(std::vector<Triangle> triangles, std::vector<Tetrahedron> cur_six_tetrahedrons, std::vector<std::array<int, 6>> cur_six_edges_rule)
{
    for(int t = 0; t < cur_six_tetrahedrons.size(); t++)
    {
        Tetrahedron cur_tetra = cur_six_tetrahedrons[t];

        cv::Point3f p01 = interpolation(cur_tetra.vertices[0], cur_tetra.vertices[1], cur_tetra.density[0], cur_tetra.density[1], ISOVALUE);
        cv::Point3f p02 = interpolation(cur_tetra.vertices[0], cur_tetra.vertices[2], cur_tetra.density[0], cur_tetra.density[2], ISOVALUE);
        cv::Point3f p03 = interpolation(cur_tetra.vertices[0], cur_tetra.vertices[3], cur_tetra.density[0], cur_tetra.density[3], ISOVALUE);
        cv::Point3f p12 = interpolation(cur_tetra.vertices[1], cur_tetra.vertices[2], cur_tetra.density[1], cur_tetra.density[2], ISOVALUE);
        cv::Point3f p23 = interpolation(cur_tetra.vertices[2], cur_tetra.vertices[3], cur_tetra.density[2], cur_tetra.density[3], ISOVALUE);
        cv::Point3f p31 = interpolation(cur_tetra.vertices[3], cur_tetra.vertices[1], cur_tetra.density[3], cur_tetra.density[1], ISOVALUE);

        Triangle tri1, tri2;
        if(cur_six_edges_rule[t] == std::array<int, 6>{0, 0, 0, 0, 0, 0})
            continue;
        else if(cur_six_edges_rule[t] == std::array<int, 6>{0, 0, 1, 0, 1, 1})
        {
            tri1.vertices.push_back(p03);
            tri1.vertices.push_back(p23);
            tri1.vertices.push_back(p31);
            triangles.push_back(tri1);
        }
        else if(cur_six_edges_rule[t] == std::array<int, 6>{0, 1, 0, 1, 1, 0})
        {
            tri1.vertices.push_back(p02);
            tri1.vertices.push_back(p12);
            tri1.vertices.push_back(p23);
            triangles.push_back(tri1);
        }
        else if(cur_six_edges_rule[t] == std::array<int, 6>{0, 1, 1, 1, 0, 1})
        {
            tri1.vertices.push_back(p02);
            tri1.vertices.push_back(p03);
            tri1.vertices.push_back(p31);
            triangles.push_back(tri1);

            tri2.vertices.push_back(p02);
            tri2.vertices.push_back(p31);
            tri2.vertices.push_back(p12);
            triangles.push_back(tri2);
        }
        else if(cur_six_edges_rule[t] == std::array<int, 6>{1, 0, 0, 1, 0, 1})
        {
            tri1.vertices.push_back(p01);
            tri1.vertices.push_back(p12);
            tri1.vertices.push_back(p31);
            triangles.push_back(tri1);
        }
        else if(cur_six_edges_rule[t] == std::array<int, 6>{1, 0, 1, 1, 1, 0})
        {
            tri1.vertices.push_back(p01);
            tri1.vertices.push_back(p03);
            tri1.vertices.push_back(p23);
            triangles.push_back(tri1);

            tri2.vertices.push_back(p01);
            tri2.vertices.push_back(p12);
            tri2.vertices.push_back(p23);
            triangles.push_back(tri2);

        }
        else if(cur_six_edges_rule[t] == std::array<int, 6>{1, 1, 0, 0, 1, 1})
        {
            tri1.vertices.push_back(p01);
            tri1.vertices.push_back(p02);
            tri1.vertices.push_back(p31);
            triangles.push_back(tri1);

            tri2.vertices.push_back(p02);
            tri2.vertices.push_back(p23);
            tri2.vertices.push_back(p31);
            triangles.push_back(tri2);
        }
        else if(cur_six_edges_rule[t] == std::array<int, 6>{1, 1, 1, 0, 0, 0})
        {
            tri1.vertices.push_back(p01);
            tri1.vertices.push_back(p02);
            tri1.vertices.push_back(p03);
            triangles.push_back(tri1);
        }
    }
}

#endif