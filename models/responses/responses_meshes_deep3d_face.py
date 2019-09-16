class MeshesDeep3DFaceResponse:
    def __init__(self, file_name='', message='', vertex_count=0, width_norm=0, height_norm=0,
                 mesh_vertex_xs=[], mesh_vertex_ys=[], mesh_vertex_zs=[], triangles=[],
                 mesh_vert_us=[], mesh_vert_vs=[],
                 canonical_vertex_xs=[], canonical_vertex_ys=[], canonical_vertex_zs=[],
                 feature_points_count=0, feature_point_xs=[], feature_point_ys=[], feature_point_zs=[]):
        self.json = {
            'FileName': file_name,
            'Message': message,
            'VertexCount': vertex_count,
            'WidthNorm': width_norm,
            'HeightNorm': height_norm,
            'MeshVertexXs': mesh_vertex_xs,
            'MeshVertexYs': mesh_vertex_ys,
            'MeshVertexZs': mesh_vertex_zs,
            "Triangles": triangles,
            "MeshVertUs": mesh_vert_us,
            "MeshVertVs": mesh_vert_vs,
            'CanonicalVertexXs': canonical_vertex_xs,
            'CanonicalVertexYs': canonical_vertex_ys,
            'CanonicalVertexZs': canonical_vertex_zs,
            "FeaturePointsCount": feature_points_count,
            "FeaturePointXs": feature_point_xs,
            "FeaturePointYs": feature_point_ys,
            "FeaturePointZs": feature_point_zs
        }
