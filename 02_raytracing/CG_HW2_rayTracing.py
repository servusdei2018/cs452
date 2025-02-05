import numpy as np
import matplotlib.pyplot as plt


# Define the scene
class Sphere:
    def __init__(self, center, radius, color, ka, kd, ks, shininess):
        self.center = np.array(center)  # The 3D coordinates of the sphere's center
        self.radius = radius  # The radius of the sphere
        self.color = np.array(color)  # The RGB color of the sphere
        self.ka = ka  # Ambient coefficient
        self.kd = kd  # Diffuse coefficient
        self.ks = ks  # Specular coefficient
        self.shininess = shininess  # he shininess factor for specular highlights


def intersect_sphere(ray_origin, ray_direction, sphere):
    # Compute the coefficients of the quadratic equation
    oc = ray_origin - sphere.center
    a = np.dot(ray_direction, ray_direction)
    b = 2.0 * np.dot(oc, ray_direction)
    c = np.dot(oc, oc) - sphere.radius**2
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        return None  # No intersection

    # Compute the distance to the intersection point
    t1 = (-b - np.sqrt(discriminant)) / (2.0 * a)
    t2 = (-b + np.sqrt(discriminant)) / (2.0 * a)

    if t1 > 0 and t2 > 0:
        return min(t1, t2)  # Return the closest intersection
    elif t1 > 0:
        return t1
    elif t2 > 0:
        return t2
    else:
        return None


def blinn_phong_shading(
    intersection_point, normal, view_direction, light_position, light_color, sphere
):
    # Compute the light direction and distance
    light_direction = light_position - intersection_point
    light_distance = np.linalg.norm(light_direction)
    light_direction /= light_distance  # Normalize the light direction
    half_vector = (light_direction + view_direction) / np.linalg.norm(
        light_direction + view_direction
    )

    # Compute the ambient component
    ambient = sphere.ka * sphere.color
    # Compute the diffuse component
    diffuse = sphere.kd * max(np.dot(normal, light_direction), 0) * sphere.color
    # Compute the specular component
    specular = (
        sphere.ks
        * (max(np.dot(normal, half_vector), 0) ** sphere.shininess)
        * light_color
    )

    # Combine all components and normalize
    color = ambient + diffuse + specular
    color = np.clip(color, 0, 1)  # Ensure color values are within [0, 1]

    return color


def render_scene():
    # Set background to grey
    image = np.zeros((image_height, image_width, 3))
    image[:] = 0.5

    # Loop over each pixel
    for y in range(image_height):
        for x in range(image_width):
            pixel_ndc = np.array([(x + 0.5) / image_width, (y + 0.5) / image_height])
            pixel_camera = np.array(
                [
                    (2 * pixel_ndc[0] - 1) * (image_plane_width / 2),
                    -(1 - 2 * pixel_ndc[1]) * (image_plane_height / 2),
                    -image_plane_dist,
                ]
            )
            ray_direction = pixel_camera - camera_position
            ray_direction /= np.linalg.norm(
                ray_direction
            )  # Normalize the ray direction

            # Find the closest intersection with a sphere
            closest_t = float("inf")
            closest_sphere = None
            for sphere in spheres:
                t = intersect_sphere(camera_position, ray_direction, sphere)
                if t is not None and t < closest_t:
                    closest_t = t
                    closest_sphere = sphere

            # If there is an intersection, shade the pixel
            if closest_sphere is not None:
                intersection_point = camera_position + closest_t * ray_direction
                normal = (intersection_point - closest_sphere.center) / np.linalg.norm(
                    intersection_point - closest_sphere.center
                )
                view_direction = (
                    -ray_direction
                )  # The view direction is opposite to the ray direction

                # Compute the color of the pixel using Blinn-Phong shading
                color = blinn_phong_shading(
                    intersection_point,
                    normal,
                    view_direction,
                    light_position,
                    light_color,
                    closest_sphere,
                )
                image[y, x] = color

    return image


# Main function
if __name__ == "__main__":
    # Define spheres
    spheres = [
        Sphere(
            center=[0, 0, -5],
            radius=1,
            color=[0, 1, 1],
            ka=0.45,
            kd=0.7,
            ks=0.5,
            shininess=32,
        ),  # Cyan
        Sphere(
            center=[2, 0, -6],
            radius=1.5,
            color=[1, 0, 1],
            ka=0.45,
            kd=0.7,
            ks=0.5,
            shininess=32,
        ),  # Magenta
    ]

    # Define light source
    light_position = np.array([5, 5, -10])
    light_intensity = 1.0
    light_color = np.array([1, 1, 1])  # White light

    # Render the scene
    image_width = 800
    image_height = 800

    # Initial pixel colors of the scene (final output image)
    pixel_colors = np.zeros((image_height, image_width, 3))

    # Define the image plane
    image_plane_height = 2.0
    aspect_ratio = image_width / image_height
    image_plane_width = aspect_ratio * image_plane_height

    # Define the camera
    camera_position = np.array([0, 0, 0])

    # Distance between the camera and the image plane
    image_plane_dist = 1.0

    # Render the scene
    image = render_scene()

    # Display the image
    plt.imshow(image)
    # plt.axis('off')
    plt.show()
