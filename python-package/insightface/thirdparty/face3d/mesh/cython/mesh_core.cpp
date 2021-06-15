/*
functions that can not be optimazed by vertorization in python.
1. rasterization.(need process each triangle)
2. normal of each vertex.(use one-ring, need process each vertex)
3. write obj(seems that it can be verctorized? anyway, writing it in c++ is simple, so also add function here. --> however, why writting in c++ is still slow?)

Author: Yao Feng 
Mail: yaofeng1995@gmail.com
*/

#include "mesh_core.h"


/* Judge whether the point is in the triangle
Method:
    http://blackpawn.com/texts/pointinpoly/
Args:
    point: [x, y] 
    tri_points: three vertices(2d points) of a triangle. 2 coords x 3 vertices
Returns:
    bool: true for in triangle
*/
bool isPointInTri(point p, point p0, point p1, point p2)
{   
    // vectors
    point v0, v1, v2;
    v0 = p2 - p0;
    v1 = p1 - p0;
    v2 = p - p0;

    // dot products
    float dot00 = v0.dot(v0); //v0.x * v0.x + v0.y * v0.y //np.dot(v0.T, v0)
    float dot01 = v0.dot(v1); //v0.x * v1.x + v0.y * v1.y //np.dot(v0.T, v1)
    float dot02 = v0.dot(v2); //v0.x * v2.x + v0.y * v2.y //np.dot(v0.T, v2)
    float dot11 = v1.dot(v1); //v1.x * v1.x + v1.y * v1.y //np.dot(v1.T, v1)
    float dot12 = v1.dot(v2); //v1.x * v2.x + v1.y * v2.y//np.dot(v1.T, v2)

    // barycentric coordinates
    float inverDeno;
    if(dot00*dot11 - dot01*dot01 == 0)
        inverDeno = 0;
    else
        inverDeno = 1/(dot00*dot11 - dot01*dot01);

    float u = (dot11*dot02 - dot01*dot12)*inverDeno;
    float v = (dot00*dot12 - dot01*dot02)*inverDeno;

    // check if point in triangle
    return (u >= 0) && (v >= 0) && (u + v < 1);
}


void get_point_weight(float* weight, point p, point p0, point p1, point p2)
{   
    // vectors
    point v0, v1, v2;
    v0 = p2 - p0; 
    v1 = p1 - p0; 
    v2 = p - p0; 

    // dot products
    float dot00 = v0.dot(v0); //v0.x * v0.x + v0.y * v0.y //np.dot(v0.T, v0)
    float dot01 = v0.dot(v1); //v0.x * v1.x + v0.y * v1.y //np.dot(v0.T, v1)
    float dot02 = v0.dot(v2); //v0.x * v2.x + v0.y * v2.y //np.dot(v0.T, v2)
    float dot11 = v1.dot(v1); //v1.x * v1.x + v1.y * v1.y //np.dot(v1.T, v1)
    float dot12 = v1.dot(v2); //v1.x * v2.x + v1.y * v2.y//np.dot(v1.T, v2)

    // barycentric coordinates
    float inverDeno;
    if(dot00*dot11 - dot01*dot01 == 0)
        inverDeno = 0;
    else
        inverDeno = 1/(dot00*dot11 - dot01*dot01);

    float u = (dot11*dot02 - dot01*dot12)*inverDeno;
    float v = (dot00*dot12 - dot01*dot02)*inverDeno;

    // weight
    weight[0] = 1 - u - v;
    weight[1] = v;
    weight[2] = u;
}


void _get_normal_core(
    float* normal, float* tri_normal, int* triangles,
    int ntri)
{
    int i, j;
    int tri_p0_ind, tri_p1_ind, tri_p2_ind;

    for(i = 0; i < ntri; i++)
    {
        tri_p0_ind = triangles[3*i];
        tri_p1_ind = triangles[3*i + 1];
        tri_p2_ind = triangles[3*i + 2];

        for(j = 0; j < 3; j++)
        {
            normal[3*tri_p0_ind + j] = normal[3*tri_p0_ind + j] + tri_normal[3*i + j];
            normal[3*tri_p1_ind + j] = normal[3*tri_p1_ind + j] + tri_normal[3*i + j];
            normal[3*tri_p2_ind + j] = normal[3*tri_p2_ind + j] + tri_normal[3*i + j];
        }
    }
}


void _rasterize_triangles_core(
    float* vertices, int* triangles, 
    float* depth_buffer, int* triangle_buffer, float* barycentric_weight,
    int nver, int ntri,
    int h, int w)
{
    int i;
    int x, y, k;
    int tri_p0_ind, tri_p1_ind, tri_p2_ind;
    point p0, p1, p2, p;
    int x_min, x_max, y_min, y_max;
    float p_depth, p0_depth, p1_depth, p2_depth;
    float weight[3];

    for(i = 0; i < ntri; i++)
    {
        tri_p0_ind = triangles[3*i];
        tri_p1_ind = triangles[3*i + 1];
        tri_p2_ind = triangles[3*i + 2];

        p0.x = vertices[3*tri_p0_ind]; p0.y = vertices[3*tri_p0_ind + 1]; p0_depth = vertices[3*tri_p0_ind + 2];
        p1.x = vertices[3*tri_p1_ind]; p1.y = vertices[3*tri_p1_ind + 1]; p1_depth = vertices[3*tri_p1_ind + 2];
        p2.x = vertices[3*tri_p2_ind]; p2.y = vertices[3*tri_p2_ind + 1]; p2_depth = vertices[3*tri_p2_ind + 2];
        
        x_min = max((int)ceil(min(p0.x, min(p1.x, p2.x))), 0);
        x_max = min((int)floor(max(p0.x, max(p1.x, p2.x))), w - 1);
      
        y_min = max((int)ceil(min(p0.y, min(p1.y, p2.y))), 0);
        y_max = min((int)floor(max(p0.y, max(p1.y, p2.y))), h - 1);

        if(x_max < x_min || y_max < y_min)
        {
            continue;
        }

        for(y = y_min; y <= y_max; y++) //h
        {
            for(x = x_min; x <= x_max; x++) //w
            {
                p.x = x; p.y = y;
                if(p.x < 2 || p.x > w - 3 || p.y < 2 || p.y > h - 3 || isPointInTri(p, p0, p1, p2))
                {
                    get_point_weight(weight, p, p0, p1, p2);
                    p_depth = weight[0]*p0_depth + weight[1]*p1_depth + weight[2]*p2_depth;

                    if((p_depth > depth_buffer[y*w + x]))
                    {
                        depth_buffer[y*w + x] = p_depth;
                        triangle_buffer[y*w + x] = i;
                        for(k = 0; k < 3; k++)
                        {
                            barycentric_weight[y*w*3 + x*3 + k] = weight[k];
                        }
                    }
                }
            }
        }
    }
}


void _render_colors_core(
    float* image, float* vertices, int* triangles, 
    float* colors, 
    float* depth_buffer,
    int nver, int ntri,
    int h, int w, int c)
{
    int i;
    int x, y, k;
    int tri_p0_ind, tri_p1_ind, tri_p2_ind;
    point p0, p1, p2, p;
    int x_min, x_max, y_min, y_max;
    float p_depth, p0_depth, p1_depth, p2_depth;
    float p_color, p0_color, p1_color, p2_color;
    float weight[3];

    for(i = 0; i < ntri; i++)
    {
        tri_p0_ind = triangles[3*i];
        tri_p1_ind = triangles[3*i + 1];
        tri_p2_ind = triangles[3*i + 2];

        p0.x = vertices[3*tri_p0_ind]; p0.y = vertices[3*tri_p0_ind + 1]; p0_depth = vertices[3*tri_p0_ind + 2];
        p1.x = vertices[3*tri_p1_ind]; p1.y = vertices[3*tri_p1_ind + 1]; p1_depth = vertices[3*tri_p1_ind + 2];
        p2.x = vertices[3*tri_p2_ind]; p2.y = vertices[3*tri_p2_ind + 1]; p2_depth = vertices[3*tri_p2_ind + 2];
        
        x_min = max((int)ceil(min(p0.x, min(p1.x, p2.x))), 0);
        x_max = min((int)floor(max(p0.x, max(p1.x, p2.x))), w - 1);
      
        y_min = max((int)ceil(min(p0.y, min(p1.y, p2.y))), 0);
        y_max = min((int)floor(max(p0.y, max(p1.y, p2.y))), h - 1);

        if(x_max < x_min || y_max < y_min)
        {
            continue;
        }

        for(y = y_min; y <= y_max; y++) //h
        {
            for(x = x_min; x <= x_max; x++) //w
            {
                p.x = x; p.y = y;
                if(p.x < 2 || p.x > w - 3 || p.y < 2 || p.y > h - 3 || isPointInTri(p, p0, p1, p2))
                {
                    get_point_weight(weight, p, p0, p1, p2);
                    p_depth = weight[0]*p0_depth + weight[1]*p1_depth + weight[2]*p2_depth;

                    if((p_depth > depth_buffer[y*w + x]))
                    {
                        for(k = 0; k < c; k++) // c
                        {   
                            p0_color = colors[c*tri_p0_ind + k];
                            p1_color = colors[c*tri_p1_ind + k];
                            p2_color = colors[c*tri_p2_ind + k]; 

                            p_color = weight[0]*p0_color + weight[1]*p1_color + weight[2]*p2_color;
                            image[y*w*c + x*c + k] = p_color;
                        }

                        depth_buffer[y*w + x] = p_depth;
                    }
                }
            }
        }
    }
}


void _render_texture_core(
    float* image, float* vertices, int* triangles, 
    float* texture, float* tex_coords, int* tex_triangles, 
    float* depth_buffer,
    int nver, int tex_nver, int ntri, 
    int h, int w, int c, 
    int tex_h, int tex_w, int tex_c, 
    int mapping_type)
{
    int i;
    int x, y, k;
    int tri_p0_ind, tri_p1_ind, tri_p2_ind;
    int tex_tri_p0_ind, tex_tri_p1_ind, tex_tri_p2_ind;
    point p0, p1, p2, p;
    point tex_p0, tex_p1, tex_p2, tex_p;
    int x_min, x_max, y_min, y_max;
    float weight[3];
    float p_depth, p0_depth, p1_depth, p2_depth;
    float xd, yd;
    float ul, ur, dl, dr;
    for(i = 0; i < ntri; i++)
    {
        // mesh
        tri_p0_ind = triangles[3*i];
        tri_p1_ind = triangles[3*i + 1];
        tri_p2_ind = triangles[3*i + 2];

        p0.x = vertices[3*tri_p0_ind]; p0.y = vertices[3*tri_p0_ind + 1]; p0_depth = vertices[3*tri_p0_ind + 2];
        p1.x = vertices[3*tri_p1_ind]; p1.y = vertices[3*tri_p1_ind + 1]; p1_depth = vertices[3*tri_p1_ind + 2];
        p2.x = vertices[3*tri_p2_ind]; p2.y = vertices[3*tri_p2_ind + 1]; p2_depth = vertices[3*tri_p2_ind + 2];
       
        // texture
        tex_tri_p0_ind = tex_triangles[3*i];
        tex_tri_p1_ind = tex_triangles[3*i + 1];
        tex_tri_p2_ind = tex_triangles[3*i + 2];

        tex_p0.x = tex_coords[3*tex_tri_p0_ind]; tex_p0.y = tex_coords[3*tri_p0_ind + 1];
        tex_p1.x = tex_coords[3*tex_tri_p1_ind]; tex_p1.y = tex_coords[3*tri_p1_ind + 1];
        tex_p2.x = tex_coords[3*tex_tri_p2_ind]; tex_p2.y = tex_coords[3*tri_p2_ind + 1];


        x_min = max((int)ceil(min(p0.x, min(p1.x, p2.x))), 0);
        x_max = min((int)floor(max(p0.x, max(p1.x, p2.x))), w - 1);
      
        y_min = max((int)ceil(min(p0.y, min(p1.y, p2.y))), 0);
        y_max = min((int)floor(max(p0.y, max(p1.y, p2.y))), h - 1);


        if(x_max < x_min || y_max < y_min)
        {
            continue;
        }

        for(y = y_min; y <= y_max; y++) //h
        {
            for(x = x_min; x <= x_max; x++) //w
            {
                p.x = x; p.y = y;
                if(p.x < 2 || p.x > w - 3 || p.y < 2 || p.y > h - 3 || isPointInTri(p, p0, p1, p2))
                {
                    get_point_weight(weight, p, p0, p1, p2);
                    p_depth = weight[0]*p0_depth + weight[1]*p1_depth + weight[2]*p2_depth;
                    
                    if((p_depth > depth_buffer[y*w + x]))
                    {
                        // -- color from texture
                        // cal weight in mesh tri
                        get_point_weight(weight, p, p0, p1, p2);
                        // cal coord in texture
                        tex_p = tex_p0*weight[0] + tex_p1*weight[1] + tex_p2*weight[2];
                        tex_p.x = max(min(tex_p.x, float(tex_w - 1)), float(0)); 
                        tex_p.y = max(min(tex_p.y, float(tex_h - 1)), float(0)); 

                        yd = tex_p.y - floor(tex_p.y);
                        xd = tex_p.x - floor(tex_p.x);
                        for(k = 0; k < c; k++)
                        {
                            if(mapping_type==0)// nearest
                            {   
                                image[y*w*c + x*c + k] = texture[int(round(tex_p.y))*tex_w*tex_c + int(round(tex_p.x))*tex_c + k];
                            }
                            else//bilinear interp
                            { 
                                ul = texture[(int)floor(tex_p.y)*tex_w*tex_c + (int)floor(tex_p.x)*tex_c + k];
                                ur = texture[(int)floor(tex_p.y)*tex_w*tex_c + (int)ceil(tex_p.x)*tex_c + k];
                                dl = texture[(int)ceil(tex_p.y)*tex_w*tex_c + (int)floor(tex_p.x)*tex_c + k];
                                dr = texture[(int)ceil(tex_p.y)*tex_w*tex_c + (int)ceil(tex_p.x)*tex_c + k];

                                image[y*w*c + x*c + k] = ul*(1-xd)*(1-yd) + ur*xd*(1-yd) + dl*(1-xd)*yd + dr*xd*yd;
                            }

                        }

                        depth_buffer[y*w + x] = p_depth;
                    } 
                }
            }
        }
    }
}



// ------------------------------------------------- write
// obj write
// Ref: https://github.com/patrikhuber/eos/blob/master/include/eos/core/Mesh.hpp
void _write_obj_with_colors_texture(string filename, string mtl_name, 
    float* vertices, int* triangles, float* colors, float* uv_coords,
    int nver, int ntri, int ntexver)
{
    int i;

    ofstream obj_file(filename.c_str());

    // first line of the obj file: the mtl name
    obj_file << "mtllib " << mtl_name << endl;
    
    // write vertices 
    for (i = 0; i < nver; ++i) 
    {
        obj_file << "v " << vertices[3*i] << " " << vertices[3*i + 1] << " " << vertices[3*i + 2] << colors[3*i] << " " << colors[3*i + 1] << " " << colors[3*i + 2] <<  endl;
    }

    // write uv coordinates
    for (i = 0; i < ntexver; ++i) 
    {
        //obj_file << "vt " << uv_coords[2*i] << " " << (1 - uv_coords[2*i + 1]) << endl;
        obj_file << "vt " << uv_coords[2*i] << " " << uv_coords[2*i + 1] << endl;
    }

    obj_file << "usemtl FaceTexture" << endl;
    // write triangles
    for (i = 0; i < ntri; ++i) 
    {
        // obj_file << "f " << triangles[3*i] << "/" << triangles[3*i] << " " << triangles[3*i + 1] << "/" << triangles[3*i + 1] << " " << triangles[3*i + 2] << "/" << triangles[3*i + 2] << endl;
        obj_file << "f " << triangles[3*i + 2] << "/" << triangles[3*i + 2] << " " << triangles[3*i + 1] << "/" << triangles[3*i + 1] << " " << triangles[3*i] << "/" << triangles[3*i] << endl;
    }

}
