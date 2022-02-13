import numpy as np
cimport numpy as np
from libcpp.string cimport string

# use the Numpy-C-API from Cython
np.import_array()

# cdefine the signature of our c function
cdef extern from "mesh_core.h":
    void _rasterize_triangles_core(
        float* vertices, int* triangles, 
        float* depth_buffer, int* triangle_buffer, float* barycentric_weight,
        int nver, int ntri,
        int h, int w)

    void _render_colors_core(
        float* image, float* vertices, int* triangles, 
        float* colors, 
        float* depth_buffer,
        int nver, int ntri,
        int h, int w, int c)

    void _render_texture_core(
        float* image, float* vertices, int* triangles, 
        float* texture, float* tex_coords, int* tex_triangles, 
        float* depth_buffer,
        int nver, int tex_nver, int ntri, 
        int h, int w, int c, 
        int tex_h, int tex_w, int tex_c, 
        int mapping_type)

    void _get_normal_core(
        float* normal, float* tri_normal, int* triangles,
        int ntri)

    void _write_obj_with_colors_texture(string filename, string mtl_name, 
        float* vertices, int* triangles, float* colors, float* uv_coords,
        int nver, int ntri, int ntexver)

def get_normal_core(np.ndarray[float, ndim=2, mode = "c"] normal not None, 
                np.ndarray[float, ndim=2, mode = "c"] tri_normal not None, 
                np.ndarray[int, ndim=2, mode="c"] triangles not None, 
                int ntri
                ):
    _get_normal_core(
        <float*> np.PyArray_DATA(normal), <float*> np.PyArray_DATA(tri_normal), <int*> np.PyArray_DATA(triangles),  
        ntri)

def rasterize_triangles_core(
                np.ndarray[float, ndim=2, mode = "c"] vertices not None, 
                np.ndarray[int, ndim=2, mode="c"] triangles not None, 
                np.ndarray[float, ndim=2, mode = "c"] depth_buffer not None,
                np.ndarray[int, ndim=2, mode = "c"] triangle_buffer not None,
                np.ndarray[float, ndim=2, mode = "c"] barycentric_weight not None,
                int nver, int ntri,
                int h, int w
                ):   
    _rasterize_triangles_core(
        <float*> np.PyArray_DATA(vertices), <int*> np.PyArray_DATA(triangles),  
        <float*> np.PyArray_DATA(depth_buffer), <int*> np.PyArray_DATA(triangle_buffer), <float*> np.PyArray_DATA(barycentric_weight),
        nver, ntri,
        h, w)

def render_colors_core(np.ndarray[float, ndim=3, mode = "c"] image not None, 
                np.ndarray[float, ndim=2, mode = "c"] vertices not None, 
                np.ndarray[int, ndim=2, mode="c"] triangles not None, 
                np.ndarray[float, ndim=2, mode = "c"] colors not None, 
                np.ndarray[float, ndim=2, mode = "c"] depth_buffer not None,
                int nver, int ntri,
                int h, int w, int c
                ):   
    _render_colors_core(
        <float*> np.PyArray_DATA(image), <float*> np.PyArray_DATA(vertices), <int*> np.PyArray_DATA(triangles),  
        <float*> np.PyArray_DATA(colors), 
        <float*> np.PyArray_DATA(depth_buffer),
        nver, ntri,
        h, w, c)

def render_texture_core(np.ndarray[float, ndim=3, mode = "c"] image not None, 
                np.ndarray[float, ndim=2, mode = "c"] vertices not None, 
                np.ndarray[int, ndim=2, mode="c"] triangles not None, 
                np.ndarray[float, ndim=3, mode = "c"] texture not None, 
                np.ndarray[float, ndim=2, mode = "c"] tex_coords not None, 
                np.ndarray[int, ndim=2, mode="c"] tex_triangles not None, 
                np.ndarray[float, ndim=2, mode = "c"] depth_buffer not None,
                int nver, int tex_nver, int ntri,
                int h, int w, int c,
                int tex_h, int tex_w, int tex_c,
                int mapping_type
                ):   
    _render_texture_core(
        <float*> np.PyArray_DATA(image), <float*> np.PyArray_DATA(vertices), <int*> np.PyArray_DATA(triangles),  
        <float*> np.PyArray_DATA(texture), <float*> np.PyArray_DATA(tex_coords), <int*> np.PyArray_DATA(tex_triangles),  
        <float*> np.PyArray_DATA(depth_buffer),
        nver, tex_nver, ntri,
        h, w, c, 
        tex_h, tex_w, tex_c, 
        mapping_type)

def write_obj_with_colors_texture_core(string filename, string mtl_name, 
                np.ndarray[float, ndim=2, mode = "c"] vertices not None, 
                np.ndarray[int, ndim=2, mode="c"] triangles not None, 
                np.ndarray[float, ndim=2, mode = "c"] colors not None, 
                np.ndarray[float, ndim=2, mode = "c"] uv_coords not None, 
                int nver, int ntri, int ntexver
                ):
    _write_obj_with_colors_texture(filename, mtl_name, 
        <float*> np.PyArray_DATA(vertices), <int*> np.PyArray_DATA(triangles), <float*> np.PyArray_DATA(colors), <float*> np.PyArray_DATA(uv_coords),
        nver, ntri, ntexver)
