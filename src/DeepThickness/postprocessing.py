#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors:
* Connor Dalby, University of Glasgow
* Damiano Ferrari, University of Brescia
* Michele Svanera, University of Glasgow

Postprocessing class to manage all operations/functions relating to fixing any issues with the cortical surface maps/meshes.
"""
import pymeshlab as pml

class Post_Processing:
    """
    A class for applying post processing operations to a surface mesh. Once initialised, the following pymeshlab 
    operations can be applied to the class mesh object:
    - Whole mesh Taubin smoothing
    - Laplacian smoothing intersecting faces
    - Taubin smoothing intersecting faces
    - Count number of intersecting faces
    - Return vertices matrix
    - Return faces matrix
    
    """
    
    def __init__(self, vertices, faces):
        
        mesh_set = pml.MeshSet()
        
        mesh_set.add_mesh(pml.Mesh(vertices, faces))
        
        self.mesh = mesh_set  

    def apply_whole_mesh_taubin_smoothing(self, lambda_ = 0.5, mu = -0.53):
        """
        Apply Taubin smoothing to the whole mesh.
        """

        self.mesh.set_selection_none()
        
        self.mesh.apply_coord_taubin_smoothing(lambda_=lambda_, mu=mu)
        
        self.mesh.set_selection_none()
        

        return self  
    
    def apply_laplacian_smoothing(self, angledeg=20, iterations=5):
        """
        Apply Laplacian smoothing to only the intersecting faces + dilation of selection, 
        preserving the surface structure as much as possible.
        Max Normal Dev (deg): maximum mean normal angle displacement (degrees) from old to new faces
        Iterations: number of laplacian smooth iterations in every run
        
        """
        self.mesh.compute_selection_by_self_intersections_per_face()
        
        self.mesh.apply_selection_dilatation()
        
        self.mesh.apply_coord_laplacian_smoothing_surface_preserving(
            selection=True, angledeg=angledeg, iterations=iterations)
        
        self.mesh.set_selection_none()
        
        return self  

    def repair_non_manifold_edges(self):
        """
        Repair the non manifold edges for sucessive curvature calculations by removing faces while 
        keeping vertices count the same. Relies on the pymeshlab filter.
        """
        self.mesh.meshing_repair_non_manifold_edges(method=0) # method = remove faces
    
        return self
    
    def apply_taubin_smoothing(self, lambda_ = 0.5, mu = -0.53, stepsmoothnum = 25):
        """
        Apply Taubin smoothing to only the intersecting faces + dilation of selection, 
        preserving the surface structure as much as possible.
        lambda is the factor controlling the initial Laplacian (diffusion) smoothing
        mu is the counteracting parameter that reverses excessive shrinkage to help preserve the original shape
        
        """
        
        self.mesh.compute_selection_by_self_intersections_per_face()

        self.mesh.apply_selection_dilatation()

        self.mesh.apply_coord_taubin_smoothing(lambda_ = lambda_, mu = mu, stepsmoothnum = stepsmoothnum, selected=True)
        
        self.mesh.set_selection_none()

        return self  
        
    def get_vertices(self):
        
        return self.mesh[0].vertex_matrix() 

    def get_faces(self):
        
        return self.mesh[0].face_matrix() 

    def count_intersections(self):
        
        self.mesh.compute_selection_by_self_intersections_per_face()  
        
        intersection_count = len(self.mesh[0].face_matrix()[self.mesh[0].face_selection_array()])
    
        
        self.mesh.set_selection_none()
        
        return intersection_count
