# Marching Cubes using Tetrahedrons Method
## [Goal] Overcome an ambiguity problem of basic Marching Cubes algorithm 
- Example pointcloud 
   <table>
      <tr>
         <td><img src="./results/input/airplane1.png"/> </td>
         <td><img src="./results/input/airplane2.png"/> </td>
      </tr>
   </table>

## 1. Prerequisites
### 1.1 Dependencies
OpenCV 3.2.0, C++ 11 version

### 1.2. OpenCV Installation
Follow [OpenCV](https://docs.opencv.org/4.x/d2/de6/tutorial_py_setup_in_ubuntu.html)
- Install appropriate OpenCV version: [Here](https://sungjaeshin.github.io/O/opencv-install/).

### 1.3. Basic Marching Cubes Github
Follow this repo &rarr; [Marching Cubes](https://github.com/SungJaeShin/Marching_cubes.git)

## 2. Changing Parameters
### Parameters in "parameters.h"
```
// random grid (READ_FILE = 0) or read from file (READ_FILE = 1)
#define READ_FILE 1 

// Maximum Grid Size
#define GRID_MAX 400

// Number of Voxel  
#define NUM_VOXEL 400

// Set ISOVALUE
#define ISOVALUE 0.5
```

## 3. Descriptions
(1) save_ply.h from [https://github.com/nihaljn/marching-cubes/blob/main/src/utilities.cpp](https://github.com/nihaljn/marching-cubes/blob/main/src/utilities.cpp) \
(2) Input file format &rarr; `.ply` & `.txt` \
(3) If you don't have input files, then you can create random grid &rarr; `generate_random_grid()` in `utility.h` \
(4) Output file format &rarr; `.ply` & `.txt` \
(5) Visualized pointcloud or mesh &rarr; `viz3DMesh()` & `viz3DPoints()` in `viz_mesh.h` \
(6) Visualization python code also provided in `example` folder &rarr; `viz_ply.py` \
(7) Convert PLY format Binary to ASCII in `example` folder &rarr; `cvt_binary2ascii.py` 

## 4. Build and Run 
Clone the repository and build and run simultaneously:
```
   $ cd ${workspace}
   $ git clone https://github.com/SungJaeShin/Marching_tetrahedrons.git
   $ cd Marching_tetrahedrons
   $ sh start.sh
```

In `start.sh` file, **there must write the file (PLY or TXT) location and output file (PLY or TXT) location** !!
```
g++ ./src/main.cpp -L /usr/local/include/opencv2 -lopencv_viz -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_core -lopencv_features2d -o ./marching
./marching <INPUT_FILE_LOCATION> <OUTPUT_SAVE_LOCATION>
```

## 5. Setting Rules between Vertices and Edges !!
```

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

```

  <table>
      <tr>
         <td> Case 1 : Not Make Any Figures </td>
         <td> <img src="./descriptions/not_make_any_figures.jpg"/> </td>
      </tr>
      <tr>
         <td> Case 2 : Make Triangles </td>
         <td> <img src="./descriptions/make_triangles.jpg"/> </td>
      </tr>
      <tr> 
         <td> Case 3 : Make Squares </td>
         <td> <img src="./descriptions/make_squares.jpg"/> </td>
      </tr>
  </table>

## 6. Results



## 7. References
[1] [https://github.com/SungJaeShin/Marching_cubes.git](https://github.com/SungJaeShin/Marching_cubes.git) \
[2] [https://paulbourke.net/geometry/polygonise/](https://paulbourke.net/geometry/polygonise/) \
[3] [https://paulbourke.net/geometry/polygonise/source1.c](https://paulbourke.net/geometry/polygonise/source1.c) 
