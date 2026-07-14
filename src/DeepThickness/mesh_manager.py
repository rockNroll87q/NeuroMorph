#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors:
* Connor Dalby, University of Glasgow
* Damiano Ferrari, University of Brescia
* Michele Svanera, University of Glasgow

CorticalSurfaceMap class to manage all operations/functions relating to the cortical surface maps/meshes.
"""


import numpy as np
import nibabel as nib
import pymeshlab as pml
import os
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import KDTree
from loguru import logger
import fast_simplification
from skimage.measure import marching_cubes

from DeepThickness.postprocessing import Post_Processing

class CorticalSurfaceMap:
    """
    This class is used to deal with cortical surfaces.
    """
    def __init__(self, vertices:np.ndarray, faces:np.ndarray, vertices_values:np.ndarray):
        assert vertices.shape[1] == 3
        assert len(vertices) == len(vertices_values)
        self.vertices = vertices
        self.faces = faces
        self.vertices_values = vertices_values
    
    @classmethod
    def from_mesh(cls, path:str) -> 'CorticalSurfaceMap': 
        """
        This alternate constructor reads a mesh from a path and stores it in
        this object. We only accept the `.ply` format sice it let us save in a
        sigle compressed file both the mesh and the overlay.
        """
        assert path.endswith(".ply")
        assert os.path.exists(path) and os.path.isfile(path), f"{path} is not a file or does not exist!"
        mesh_set = pml.MeshSet()
        mesh_set.load_new_mesh(path)
        mesh = mesh_set[0]
        return cls(mesh.vertex_matrix(), mesh.face_matrix(), mesh.vertex_scalar_array())

    @classmethod
    def from_volumetric_data(cls, level_set:np.ndarray, distance_set:np.ndarray, allow_degenerate=False, smooth_thickness=False, 
                             mc_level = 1.0, norm_mult = 1.0, apply_post_processing=True, mesh_decimation_target=75000, subject_name=None) -> 'CorticalSurfaceMap':
        """
        Given the level set of a surface and the corresponding distance array (it could be both a distance set or a level set since it takes the absolute value), 
        returns the surface (vertices and faces) and the thickness associated to each vertex. MC level controls the isovalue threshold at which a surface boundary
        is set in the MC algorithm (how much to cut into the surface). If altering the mc_level, we 're-inflate' the mesh by moving all faces along the norms by
        a fixed value (norm_mult). We remove any isolate pieces and decimate the mesh to a fixed target number of vertices. A series of smoothing can be applied 
        as post processing (recommended) to improve mesh quality and reduce intersections. Finally, we interpolate the vertices with the distance set to generate a CTh as mesh
        colour overlay. If smooth_thickness is True, it smooths these vertices values. 
        
        Args:
            level_set: Volumetric level-set used to extract the isosurface.
            distance_set: Volumetric values sampled on the output mesh to
                generate thickness overlay (absolute value is used).
            allow_degenerate: Passed to marching cubes to allow degenerate
                triangles.
            smooth_thickness: If True, applies per-vertex scalar smoothing to
                the sampled overlay values.
            mc_level: Isovalue used by marching cubes.
            norm_mult: Normal displacement multiplier applied when
                `mc_level != 0`.
            apply_post_processing: If True, applies smoothing/repair steps from
                `Post_Processing`.
            mesh_decimation_target: Target number of vertices for mesh
                simplification.
            subject_name: Optional subject identifier used in warnings/asserts.
        
        """
        
        # Surface reconstruction
        coords, faces, normals, values = marching_cubes(level_set, level=mc_level, allow_degenerate=allow_degenerate)
        
        
        # Overlay
        if mc_level != 0:
            # Inflate
            mod_norms = normals * norm_mult
            coords = coords + mod_norms

        # Decimate the mesh to an exact vertex count 
        ct_map = CorticalSurfaceMap(coords, faces, np.zeros_like(coords))
        ct_map.remove_isolated_pieces()
        ct_map.fast_load_and_simplify_mesh(mesh_decimation_target=mesh_decimation_target, subject_name=subject_name)
        
        
        # Optionally apply post processing
        
        if apply_post_processing:    
            """
            To improve the quality of the mesh and reduce the number of intersections, we can optionally apply post processing to the mesh.
            This is acheived by applying:
            - Minor Taubin smoothing to the whole mesh 
            - Laplacian smoothing on intersecting faces
            - Up to 5 rounds of Taubin smoothing on intersecting faces
            """

            modified_ms = Post_Processing(vertices = ct_map.vertices, faces = ct_map.faces)  
            modified_ms.apply_whole_mesh_taubin_smoothing()  # Apply whole mesh smoothing
            modified_ms.apply_laplacian_smoothing()  # Apply laplacian smoothing        
            final_intersections =  modified_ms.count_intersections() # Set as final to skip Taubin if there are intersections at this point
            
            for i in range(5):
                
                if final_intersections == 0: # Exit loop if there are no intersections left
                    break

                modified_ms.apply_taubin_smoothing() # Apply Taubin smoothing
                final_intersections = modified_ms.count_intersections() 
            modified_ms.repair_non_manifold_edges()
            
            ct_map.vertices = modified_ms.get_vertices()
            ct_map.faces = modified_ms.get_faces()
        
            
        # Interpolate the re-inflated mesh
        ct_map.apply_from_distance_set(distance_set = distance_set, smooth_thickness = smooth_thickness, interpolation_method = 'linear')         

        return ct_map

    @classmethod
    def from_freesurfer(cls, surface_path:str, thickness_path:str, orig_path:str=None) -> 'CorticalSurfaceMap':
        """
        Build a cortical surface map from FreeSurfer surface and thickness data.

        Args:
            surface_path: Path to FreeSurfer geometry file.
            thickness_path: Path to FreeSurfer morphometry thickness file.
            orig_path: Optional path to an image whose `vox2ras_tkr` is used for
                RAS-to-voxel conversion. If None, uses internal `_get_T_orig()`.

        Returns:
            CorticalSurfaceMap: Surface converted to voxel space with thickness
            values as overlay.
        """
        # Get ras2vox matrix
        if orig_path is None:
            ras2vox = np.linalg.inv(_get_T_orig())
        else:
            ras2vox = np.linalg.inv(nib.load(orig_path).header.get_vox2ras_tkr())

        # Read surface
        coords_ras, faces = nib.freesurfer.read_geometry(surface_path)

        # Convert coordinates to vox space
        extended_coords = np.hstack([coords_ras, np.ones((len(coords_ras), 1))])
        coords_vox = np.empty_like(coords_ras)
        for i, coord in enumerate(extended_coords):
            coords_vox[i] = (ras2vox @ coord)[:-1] # elide trailing 1
        
        # Read thickness
        thickness = nib.freesurfer.io.read_morph_data(thickness_path)

        return cls(coords_vox, faces, thickness)
    
    def get_values(self):
        """
        Return mesh arrays.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
            `(vertices, faces, vertices_values)`.
        """
        return self.vertices, self.faces, self.vertices_values

    def copy(self) -> 'CorticalSurfaceMap':
        """
        Create a deep copy of this surface map.

        Returns:
            CorticalSurfaceMap: Copied mesh and overlay arrays.
        """
        return CorticalSurfaceMap(self.vertices.copy(), self.faces.copy(), self.vertices_values.copy())

    def merge(self, other:'CorticalSurfaceMap'):
        """
        Merge `other` `CorticalSurfaceMap` with this object. This is very useful
        when you load the two hemispheres of freesurfer and you want to merge
        them in a single object.

        Args:
            other: Surface map merged into this instance.

        Returns:
            None

        ## Typical usage
        >>> gt_surface = CorticalSurfaceMap.from_freesurfer(lh_surface_path, lh_thickness_path, orig_path)
        >>> gt_surface.merge(CorticalSurfaceMap.from_freesurfer(rh_surface_path, rh_thickness_path, orig_path))
        """
        self.faces = np.vstack((self.faces, other.faces + len(self.vertices)))
        self.vertices = np.vstack((self.vertices, other.vertices))
        self.vertices_values = np.hstack((self.vertices_values, other.vertices_values))
    
    def apply_from_distance_set(self, distance_set:np.ndarray, smooth_thickness=False, interpolation_method='linear'):
        """
        This method computes a cortical thickness overlay by using the given
        distance set. It uses a trilinear interpolation. `distance_set` could
        also be the level set of the opposite cortical surface since we take the
        absoulte value. If `smooth_thickness` is `True`, it smooths the vertices
        values.
        """
        interp = RegularGridInterpolator((np.arange(distance_set.shape[0]), np.arange(distance_set.shape[1]), np.arange(distance_set.shape[2])),
                                         np.abs(distance_set), method=interpolation_method,
                                         bounds_error=False, fill_value=0.0)
        self.vertices_values = interp(self.vertices)
        if smooth_thickness:
            mesh_set = pml.MeshSet()
            mesh_set.add_mesh(pml.Mesh(self.vertices, self.faces, v_scalar_array=self.vertices_values))
            mesh_set.apply_scalar_smoothing_per_vertex()
            self.vertices_values = mesh_set[0].vertex_scalar_array()

    def compute_thickness_as_distance_to_opposite_mesh(self, other:'CorticalSurfaceMap'):
        """
        This method set as the thickness map the distance to another mesh. Note
        that FreeSurfer uses as the thickness overlay the average between the 2
        distances definitions (distance from pial to white and distance from
        white to pial). This function is optimized, it takes just few seconds to
        compute all the distances between 2 meshes of 300k+ vertices each.
        """
        opposite_mesh_kdtree = KDTree(other.vertices)
        for index, vertex in enumerate(self.vertices):
            self.vertices_values[index] = opposite_mesh_kdtree.query(vertex)[0]

    def save(self, output_path:str, quality_mapper:str, minval:float, maxval:float, 
             quality_mapper_path = '/NeuroMorph/src/DeepThickness/quality_mappers/'):
        """
        This method saves this object to a path. We only accept the `.ply`
        format sice it let us save in a sigle compressed file both the mesh and
        the overlay. Quality mapper could be None, "BGR" or "BR".
        
        Args:
            output_path: Destination `.ply` file path.
            quality_mapper: Color transfer function selector: `None`, `"BR"`, or
                `"BGR"`.
            minval: Minimum scalar value for transfer-function mapping.
            maxval: Maximum scalar value for transfer-function mapping.
            quality_mapper_path: Directory containing `*.tf` transfer-function
                files.
        
        """
        if not os.path.exists(quality_mapper_path):
            logger.warning(f"Quality mapper path {quality_mapper_path} does not exist. Using default path /NeuroMorph/src/DeepThickness/quality_mappers/")
            quality_mapper_path = '/NeuroMorph/src/DeepThickness/quality_mappers/'
        assert output_path.endswith(".ply")
        assert quality_mapper is None or quality_mapper == "BR" or quality_mapper == "BGR"
        mesh_set = pml.MeshSet()
        mesh_set.add_mesh(pml.Mesh(self.vertices, self.faces, v_scalar_array=self.vertices_values))
        # If no color is selected color vertices with grey (best color to see mesh problems)
        if quality_mapper is None:
            mesh_set.compute_color_by_function_per_vertex(x="128", y="128", z="128")
        else:
            mesh_set.compute_color_from_scalar_using_transfer_function_per_vertex(minqualityval=minval, maxqualityval=maxval, tfslist="Custom Transfer Function File",csvfilename=f"{quality_mapper_path}{quality_mapper}.tf")
        mesh_set.save_current_mesh(output_path, binary=True)

    def save_as_freesurfer(self, out_surface_path:str, out_overlay_path=None):
        """
        Save mesh geometry (and optionally overlay) in FreeSurfer format.

        Args:
            out_surface_path: Output path for FreeSurfer geometry.
            out_overlay_path: Optional output path for morph data overlay.

        Returns:
            None
        """

        T_orig = _get_T_orig()
        extended_coords = np.hstack([self.vertices, np.ones((len(self.vertices), 1))])

        coords_surf = np.empty_like(self.vertices)
        for i, coord in enumerate(extended_coords):
            coords_surf[i] = (T_orig @ coord)[:-1] # elide trailing 1

        nib.freesurfer.write_geometry(out_surface_path, coords_surf, self.faces)
        if out_overlay_path is not None:
            nib.freesurfer.write_morph_data(out_overlay_path, self.vertices_values)

    def remove_isolated_pieces(self):
        """
        Remove any isolated/unconnected components of a mesh that are smaller than 10% of the diameter of the whole mesh
        """
        
        ms = pml.MeshSet()
        ms.add_mesh(pml.Mesh(self.vertices, self.faces))
        ms.meshing_remove_connected_component_by_diameter()
        self.vertices = ms[0].vertex_matrix()
        self.faces = ms[0].face_matrix()
        

    def fast_load_and_simplify_mesh(self, mesh_decimation_target=75000, subject_name=None):
        
        """
        This method rapidly simplifies a mesh to a given vertex rought target and then selectively refines its largest faces to restore geometric detail. 
        First, it iteratively applies a the fast decimation algorithm/paclage until the mesh has at most mesh_decimation_target vertices. 
        It then computes the area of each face, picks the n largest ones, and inserts a new barycenter vertex into each—splitting each chosen triangle into three smaller triangles. 
        Finally, it replaces the selected faces with these subdivided faces and updates self.vertices and self.faces, yielding a mesh that both meets the target vertex count and 
        preserves important geometric features.
        
        Args:
            mesh_decimation_target: Desired vertex count after simplification and
                face subdivision.
            subject_name: Optional subject identifier used in warnings/asserts.
        """
        
        no_of_orig_vertices = self.vertices.shape[0]
        simplification_vertices_shape = no_of_orig_vertices
        fast_target = mesh_decimation_target - 100
        
        if no_of_orig_vertices <= mesh_decimation_target:
            logger.warning(f"Subject {subject_name}: Mesh already has {no_of_orig_vertices} vertices, which is less than or equal to the target of {mesh_decimation_target}. \
                Check T1w image and predicted levelset for potential issues.")
            return self
        
        while simplification_vertices_shape > mesh_decimation_target:
            # Use fast simplification to reduce the number of self.vertices and faces close to target
            simplification_vertices, simplification_faces = fast_simplification.simplify(self.vertices, self.faces, 1-fast_target/no_of_orig_vertices,return_collapses=False)
            simplification_vertices_shape = simplification_vertices.shape[0]
            fast_target -= 15
            
        assert simplification_vertices_shape <= mesh_decimation_target, f"The fast decimation produced vertices ({simplification_vertices_shape}) greater than the target"
        n = mesh_decimation_target - simplification_vertices_shape
        
        # Use barycenter to subdivide faces
        # Extract the three vertices for each face using advanced indexing.
        A = simplification_vertices[simplification_faces[:, 0]]  # Get the first vertex of each face; shape: (num_faces, 3)
        B = simplification_vertices[simplification_faces[:, 1]]  # Get the second vertex of each face; shape: (num_faces, 3)
        C = simplification_vertices[simplification_faces[:, 2]]  # Get the third vertex of each face; shape: (num_faces, 3)
        
        # Compute two edge vectors for each face.
        edge1 = B - A  # Vector from vertex A to B; shape: (num_faces, 3)
        edge2 = C - A  # Vector from vertex A to C; shape: (num_faces, 3)
        
        # Compute the cross product of the two edge vectors for each face.
        cross_prod = np.cross(edge1, edge2)  # shape: (num_faces, 3)
        
        # Calculate the area of each face: area = 0.5 * ||cross_prod||
        areas = 0.5 * np.linalg.norm(cross_prod, axis=1)  # shape: (num_faces,)
        
        # Get the indices of the faces sorted in descending order based on their area.
        # np.argsort returns the indices that would sort the array.
        sorted_indices = np.argsort(-areas)  # sort in descending order
        
        # Select the first n indices from the sorted array.
        face_indices = sorted_indices[:n]
        
        # Ensure face_indices is a NumPy array
        face_indices = np.asarray(face_indices)
        
        # 1. Extract the selected faces from simplification_faces (each face is a triplet of vertex indices)
        selected_faces = simplification_faces[face_indices]  # shape: (k, 3), where k is the number of faces to subdivide

        # 2. Compute the barycenters for these faces in a vectorized way.
        # For each face, the barycenter is the mean of its three vertices.
        barycenters = np.mean(simplification_vertices[selected_faces], axis=1)  # shape: (k, 3)
        
        # 3. Append the new barycenter vertices to the original vertex array.
        vertices_new = np.vstack([simplification_vertices, barycenters])
        
        # The new vertices get sequential indices starting at simplification_vertices.shape[0]
        num_orig_vertices = simplification_vertices.shape[0]
        new_vertex_indices = np.arange(num_orig_vertices, num_orig_vertices + selected_faces.shape[0])
        
        # 4. For each selected face, extract the three vertex indices.
        # This yields three arrays, each of shape (k,)
        a = selected_faces[:, 0]
        b = selected_faces[:, 1]
        c = selected_faces[:, 2]
        
        # 5. Create the three new faces for each original face in a vectorized manner.
        # The new faces are:
        #   face1: [a, b, barycenter]
        #   face2: [b, c, barycenter]
        #   face3: [c, a, barycenter]
        face1 = np.stack([a, b, new_vertex_indices], axis=1)  # shape: (k, 3)
        face2 = np.stack([b, c, new_vertex_indices], axis=1)  # shape: (k, 3)
        face3 = np.stack([c, a, new_vertex_indices], axis=1)  # shape: (k, 3)
        
        # Concatenate the new faces into one array.
        new_faces = np.concatenate([face1, face2, face3], axis=0)  # shape: (3*k, 3)
        
        # 6. Remove the original subdivided faces from simplification_faces.
        # Sorting in descending order ensures that deletion does not alter subsequent indices.
        face_indices_sorted = np.sort(face_indices)[::-1]
        F_remaining = np.delete(simplification_faces, face_indices_sorted, axis=0)
        
        # 7. Combine the remaining faces with the new subdivided faces.
        faces_new = np.vstack([F_remaining, new_faces])        

        # Logging the result of decimation
        assert vertices_new.shape[0] == mesh_decimation_target, f'Warning: {subject_name} decimation failed to reach mesh_decimation_target. Decimated to {vertices_new.shape[0]} vertices from {no_of_orig_vertices}.'
                
        self.vertices = vertices_new  # Update vertices with simplified vertices
        self.faces = faces_new             # Update faces with simplified faces
        
        return self
    

    def get_curvature_mesh(self, save_path=None):
        """
        A wrapper function to utilise the pymeshlab filter to calculate the mean curvature of the mesh
        and assign each vertex a curvature value which can be visualised as a colour.
        Optionally save the mesh if a save_path is provided.
        
        Args:
            save_path: Optional path to save the curvature-colored mesh.
        """
        
        mesh_set = pml.MeshSet()
        mesh_set.add_mesh(pml.Mesh(self.vertices, self.faces))
        mesh_set.compute_curvature_and_color_apss_per_vertex(curvaturetype = 'ApproxMean')
        if save_path is not None:
            mesh_set.save_current_mesh(save_path, binary=True)

    def get_average_curvature(self):
        """
        A wrapper function to utilise the pymeshlab filter to calculate the mean curvature of the mesh
        and return the average clamped per-vertex curvature.

        Returns:
            float: Mean of vertex curvature values clipped to
            `[-sqrt(2), +sqrt(2)]`.
        """
        
        mesh_set = pml.MeshSet()
        mesh_set.add_mesh(pml.Mesh(self.vertices, self.faces))

        mesh_set.apply_filter('compute_scalar_by_discrete_curvature_per_vertex',
                        curvaturetype=0)
        curv = mesh_set[0].vertex_scalar_array()
        curv_clamped = np.clip(curv, -np.sqrt(2), + np.sqrt(2)) # 1.41mm cap used by FS 

        return np.mean(curv_clamped)
    
    def get_surface_area(self):
        """
        A wrapper function to utilise the pymeshlab filter to calculate the total surface area of the mesh
        as a single value.

        Returns:
            float: Total mesh surface area.
        """
        mesh_set = pml.MeshSet()
        mesh_set.add_mesh(pml.Mesh(self.vertices, self.faces))
        surface_area = mesh_set.get_geometric_measures()['surface_area']
        return surface_area


    def extract_mesh_metrics(self, surface_type=None):
        
        """
        From the created mesh read the CTh values as the already loaded vertices values and 
        then calculate the surface area and curvature for the mesh.
        
        Args:
            surface_type (str): the surface type to prefix the subject results 
        
        Returns:
            dict: Dictionary containing mean predicted cortical thickness,
            surface area, and curvature for this mesh.
        
        """
        surface_prefix = f'{surface_type.capitalize()}_' if surface_type is not None else "" 
        
        subject_results = {
            f"{surface_prefix}Mean_Predicted_CTh": self.vertices_values.mean(),
            f"{surface_prefix}Mesh_Predicted_Surface_Area": self.get_surface_area(),
            f"{surface_prefix}Mesh_Predicted_Curvature": self.get_average_curvature(),
        }
        
        return subject_results

def _get_T_orig():
    return np.array([[-1.,  0., 0.,  128.],
                     [ 0.,  0., 1., -128.],
                     [ 0., -1., 0.,  128.],
                     [ 0.,  0., 0.,    1.]], dtype=np.float32)
