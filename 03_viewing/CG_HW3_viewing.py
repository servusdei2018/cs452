import numpy as np
import matplotlib.pyplot as plt

# Create a 3D cube
def create_cube():
    vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # Back face
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]       # Front face
    ])
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Back face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Front face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
    ]
    return vertices, edges

# Camera Transformation
def camera_transform(vertices, eye, gaze, up):
    # homogeneous coordinates (w, u, v)
    w = -gaze / np.linalg.norm(gaze)
    up_proj = up - np.dot(up, w) * w
    u = np.cross(up_proj, w)
    u = u / np.linalg.norm(u)
    v = np.cross(w, u)

    R = np.array([u, v, w]).T
    t = -eye

    #  T = [ R  t ]
    #      [ 0  1 ]
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.dot(R,t)

    homogeneous_vertices = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
    transformed_homogeneous = np.dot(homogeneous_vertices, T.T)

    # back to cartesian
    transformed_vertices = transformed_homogeneous[:, :3]

    return transformed_vertices

# Projection Transformation
def project_vertices(vertices, projection_type="perspective", near=1, far=10, fov=np.pi/4, aspect=1.0):
    """
    Apply a projection transformation to 3D vertices.
    - perspective: applies a perspective projection.
    - orthographic: applies an orthographic projection.
    """

    if projection_type == "perspective":
        f = 1 / np.tan(fov / 2)
        projection_matrix = np.array([
            [f/aspect, 0, 0, 0],
            [0, -f, 0, 0],
            [0, 0, (far+near)/(far-near), far*near/(near-far)],
            [0, 0, 1, 0]
        ])

    elif projection_type == "orthographic":
        right, left = 1, -1
        top, bottom = 1, -1

        projection_matrix = np.array([
            [-1/(right-left), 0, 0, -(right+left)/(right-left)],
            [0, 1/(top-bottom), 0, -(top+bottom)/(top-bottom)],
            [0, 0, -1/(far-near), -(far+near)/(far-near)],
            [0, 0, 0, 1]
        ])

    homogeneous_vertices = np.column_stack([vertices, np.ones(vertices.shape[0])])
    projected_vertices = homogeneous_vertices.dot(projection_matrix.T)
    projected_vertices /= projected_vertices[:, 3][:, np.newaxis]

    return projected_vertices[:, :3]

# Viewport Transformation
def viewport_transform(vertices, width, height):
    viewport_matrix = np.array([
        [width/2, 0, 0, width/2],
        [0, height/2, 0, height/2],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    homogeneous_vertices = np.column_stack([vertices, np.ones(vertices.shape[0])])
    viewport_vertices = homogeneous_vertices.dot(viewport_matrix.T)

    return viewport_vertices[:, :2]

# Render the scene
def render_scene(vertices, edges, ax, **kwargs):
    for edge in edges:
        v1, v2 = vertices[edge[0]], vertices[edge[1]]
        ax.plot([v1[0], v2[0]], [v1[1], v2[1]], **kwargs)

# Main function
def main():
    # Scene setup
    vertices, edges = create_cube()
    eye = np.array([0.5, .5, -3])  # Camera at the origin
    gaze = np.array([-1, -1, -5])  # Looking towards the cube
    up = np.array([0, 1, 0])  # Up is along +Y-axis

    # Camera transformation
    transformed_vertices = camera_transform(vertices, eye, gaze, up)

    # Projection transformations
    perspective_vertices = project_vertices(transformed_vertices, "perspective", near=1, far=10, fov=np.pi/4, aspect=800/600)

    orthographic_vertices = project_vertices(transformed_vertices, "orthographic", near=1, far=10)

    # Viewport transformation
    viewport_width, viewport_height = 1920, 1080
    persp_2d = viewport_transform(perspective_vertices, viewport_width, viewport_height)
    ortho_2d = viewport_transform(orthographic_vertices, viewport_width, viewport_height)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].set_title("Perspective Projection")
    axes[1].set_title("Orthographic Projection")

    render_scene(persp_2d, edges, axes[0], color="blue", marker="o")
    render_scene(ortho_2d, edges, axes[1], color="red", marker="o")

    for ax in axes:
        ax.set_aspect('equal')

    plt.show()

if __name__ == "__main__":
    main()
