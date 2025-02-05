import numpy as np
import matplotlib.pyplot as plt


# Using edge method, c.f. https://www.scratchapixel.com/lessons/3d-basic-rendering/rasterization-practical-implementation/rasterization-stage.html
def is_inside_triangle(p, v1, v2, v3):
    d = [
        (p[0] - v3[0]) * (v1[1] - v3[1]) - (v1[0] - v3[0]) * (p[1] - v3[1]),
        (p[0] - v1[0]) * (v2[1] - v1[1]) - (v2[0] - v1[0]) * (p[1] - v1[1]),
        (p[0] - v2[0]) * (v3[1] - v2[1]) - (v3[0] - v2[0]) * (p[1] - v2[1]),
    ]
    signs = np.sign(d)
    return not (1 in signs and -1 in signs)


# Rasterize the triangle using barycentric coordinates
def rasterization(image: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    h, w = image.shape
    v0, v1, v2 = triangle

    for y in range(h):
        for x in range(w):
            if is_inside_triangle([x, y], v0, v1, v2):
                image[y, x] = 1

    return image


# Supersample and rasterize again for higher-quality rendering
def supersample_rasterization(
    image: np.ndarray, triangle: np.ndarray, factor: int
) -> np.ndarray:
    h, w = image.shape
    supersampled_image = np.zeros((h * factor, w * factor))

    v0, v1, v2 = triangle
    v0s, v1s, v2s = v0 * factor, v1 * factor, v2 * factor

    for y in range(h * factor):
        for x in range(w * factor):
            if is_inside_triangle([x, y], v0s, v1s, v2s):
                supersampled_image[y, x] = 1

    downsampled_image = np.zeros((h, w))

    for y in range(h):
        for x in range(w):
            downsampled_image[y, x] = np.mean(
                supersampled_image[
                    y * factor : (y + 1) * factor, x * factor : (x + 1) * factor
                ]
            )

    return downsampled_image


if __name__ == "__main__":
    triangle_vertices_1 = np.array([[22, 61], [21, 77], [96, 66]])
    triangle_vertices_2 = np.array([[21, 20], [46, 75], [88, 68]])
    triangle_vertices_3 = np.array([[5, 33], [31, 75], [45, 24]])
    triangle_vertices_4 = np.array([[10, 10], [10, 90], [90, 50]])

    # Generate image frame
    frame = np.zeros((100, 100))

    # Rasterize without supersampling
    image = rasterization(frame, triangle_vertices_1)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Rasterized Triangle")
    plt.imshow(image, cmap="gray")

    # Rasterize with supersampling
    supersampled_image = supersample_rasterization(frame, triangle_vertices_1, factor=4)

    plt.subplot(1, 2, 2)
    plt.title("Supersampled Rasterized Triangle (factor = 4)")
    plt.imshow(supersampled_image, cmap="gray")

    plt.show()
