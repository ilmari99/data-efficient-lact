import os
import numpy as np
import cv2
import sys
import random
import time
import hashlib
import threading
import scipy.spatial
import scipy.ndimage
from pathlib import Path
from multiprocessing.pool import ThreadPool

# CREDIT: https://github.com/99991/HTC2022-TUD-HHU-version-1

def hash128(seeds):
    data = hashlib.md5(str(seeds).encode("utf-8") + b"#4").digest()
    return int.from_bytes(data, byteorder="little")

def hash_randrange(seeds, start, stop=None):
    print(seeds)
    print(start)
    print(stop)
    if stop is None:
        stop = start
        start = 0
    try:
        return start + hash128(seeds) % (stop - start)
    except:
        # This happens if the user isn't in the correct directory
        raise ValueError("You must run this script in the correct directory")

def hash_rand(seeds, a, b):
    try:
        return a + (b - a) * hash128(seeds) / 2**128
    except:
        raise ValueError("You must run this script in the correct directory")

def hash_choice(seeds, values):
    return values[hash_randrange(seeds, len(values))]

def nonzero(values):
    return values[values != 0]

def load_things():
    things = []

    colors = [(235, 0, 0), (0, 157, 9), (134, 0, 210), (221, 140, 30), (128, 160, 255), (107, 84, 255), (44, 176, 168), (169, 176, 44), (163, 163, 163)]

    background_mask = cv2.imread("background_mask.png", cv2.IMREAD_GRAYSCALE) == 255

    for path in sorted(Path("things").glob("*.png")):
        # Find connected components in image
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        num_labels, labels = cv2.connectedComponents(np.uint8(image > 0))

        color_image = np.stack([image] * 3, axis=2)

        for label in range(num_labels):
            # Mask of connected component
            mask = label == labels

            # If connected component does not overlap background mask
            if not np.any(background_mask[mask]):
                # Colorize connected component
                color = colors[label % len(colors)]
                color_image[mask] = color

                # Crop component
                x0 = np.min(nonzero(np.argmax(mask, axis=1)))
                y0 = np.min(nonzero(np.argmax(mask, axis=0)))
                x1 = mask.shape[1] - 1 - np.min(nonzero(np.argmax(mask[:, ::-1], axis=1)))
                y1 = mask.shape[0] - 1 - np.min(nonzero(np.argmax(mask[::-1, :], axis=0)))

                thing = image[y0:y1, x0:x1].copy()
                thing[~mask[y0:y1, x0:x1]] = 0

                things.append(thing.astype(np.float32) / 255.0)

        # Show colorized image
        #cv2.imshow("image", color_image);cv2.waitKey(100)

    # Show things
    #for thing in things: cv2.imshow("thing", thing);cv2.waitKey(100)

    return things

def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def make_synthetic_disk(params):
    center_x, center_y, radius, d_radius, c0, c1, c2, c3 = params

    x, y = np.ogrid[-1:1:512j, -1:1:512j]

    distance = np.hypot(x - center_x, y - center_y)

    # 3rd degree polynomial to fit brightness of disk
    brightness = c0 + c1 * distance + c2 * distance**2 + c3 * distance**3

    synthetic = brightness * smoothstep(radius + d_radius, radius, distance)

    return synthetic

def make_voronoi(
    allowed_mask,
    points,
    corner_smoothing,
    border_radius,
):
    h, w = allowed_mask.shape

    y, x = np.mgrid[-1:1:1j*h, -1:1:1j*w]

    xy = np.column_stack([x.ravel(), y.ravel()])

    d, _ = scipy.spatial.KDTree(points).query(xy, k=1)

    d = d.reshape(h, w)

    kernel = np.float64([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
    ])

    d = scipy.ndimage.correlate(d, kernel, mode="nearest")

    mask = d > 0

    mask[~allowed_mask] = 0

    d = scipy.ndimage.distance_transform_edt(mask)

    d = scipy.ndimage.uniform_filter(d, size=corner_smoothing)

    d = smoothstep(border_radius, border_radius + 1, d)

    return d

def mul_affine(A, B):
    # Multiply affine 2-by-3 transformation matrices A and B
    C = np.zeros((2, 3), dtype=np.float32)
    C[0, 0] = A[0, 0] * B[0, 0] + A[0, 1] * B[1, 0]
    C[0, 1] = A[0, 0] * B[0, 1] + A[0, 1] * B[1, 1]
    C[0, 2] = A[0, 0] * B[0, 2] + A[0, 1] * B[1, 2] + A[0, 2]
    C[1, 0] = A[1, 0] * B[0, 0] + A[1, 1] * B[1, 0]
    C[1, 1] = A[1, 0] * B[0, 1] + A[1, 1] * B[1, 1]
    C[1, 2] = A[1, 0] * B[0, 2] + A[1, 1] * B[1, 2] + A[1, 2]
    return C

def warp_image_without_cropping(image, M):
    h, w = image.shape
    # Compute new bounds for image given 2-by-3 affine transformation matrix M
    corners = np.array([[[0, 0], [w, 0], [w, h], [0, h]]], dtype=np.float32)
    new_corners = cv2.transform(corners, M)[0]
    x_min, y_min = new_corners.min(axis=0)
    x_max, y_max = new_corners.max(axis=0)
    # Compute translation matrix to center image in new bounds
    T = np.array([[1, 0, -x_min], [0, 1, -y_min]], dtype=np.float32)
    M = mul_affine(T, M)
    # Compute new image dimensions
    w_new = int(np.ceil(x_max - x_min))
    h_new = int(np.ceil(y_max - y_min))
    # Apply affine transformation to image
    image_warped = cv2.warpAffine(image, M, (w_new, h_new))
    return image_warped

def generate_image(i, width, height, things):
    # params for data/htc2022_solid_disc_full_recon_fbp.png
    best_params = [0.010696374, -0.004857074, 0.912876, 0.015132266, 0.66815907, -0.068718374, 0.055300053, 0.30111307]

    params = np.array(best_params)
    jitter = 0.02
    # Jitter center
    params[0] += hash_rand([i, "center_x"], -jitter, jitter)
    params[1] += hash_rand([i, "center_y"], -jitter, jitter)
    # Jitter radius
    params[2] += hash_rand([i, "radius"], -jitter, jitter)
    # Jitter polynomial coefficients for radial brightness
    params[3] += hash_rand([i, "coeff3"], -0.001, 0.001)
    params[4] += hash_rand([i, "coeff4"], -0.01, 0.01)
    params[5] += hash_rand([i, "coeff5"], -0.01, 0.01)
    params[6] += hash_rand([i, "coeff6"], -0.01, 0.01)
    params[7] += hash_rand([i, "coeff7"], -0.01, 0.01)

    image = make_synthetic_disk(params)

    use_voronoi = hash_rand([i, "mask_dilation"], 0.0, 1.0) < 0.2

    if use_voronoi:
        mask_dilation = hash_randrange([i, "mask_dilation"], 10, 80)
        allowed_mask = np.uint8((image < 0.5) * 255)
        allowed_mask = cv2.blur(allowed_mask, (mask_dilation, mask_dilation)) < 0.99

        circle_radius = 0.7
        num_points = hash_randrange([i, "num_points"], 10, 20)
        points = np.array([
            (
                hash_rand([i, j, "x"], -0.7, 0.7),
                hash_rand([i, j, "y"], -0.7, 0.7),
            )
            for j in range(num_points)
        ])
        corner_smoothing = hash_randrange([i, "corner_smoothing"], 5, 50)
        border_radius = hash_randrange([i, "border_radius"], 3, 15) + corner_smoothing // 5

        voronoi = make_voronoi(
            allowed_mask,
            points,
            corner_smoothing,
            border_radius)

        brightness = hash_rand([i, "brightness"], 0.9, 1.0)

        image *= (1 - brightness * voronoi)

        return image

    disallowed_mask = (image == 0) * 1.0

    # Increase disallowed border of circle by a few pixels
    ksize = hash_randrange([i, "ksize"], 2, 30)
    disallowed_mask = cv2.blur(disallowed_mask, (ksize, ksize)) > 1e-5

    randomize_things = hash_choice([i, "randomize_things"], [False, True])
    randomize_transform = hash_choice([i, "randomize_transform"], [False, True])
    randomize_brightness = hash_choice([i, "randomize_brightness"], [False, True])

    num_things = hash_randrange([i, "num_things"], 1, 100)

    for i_thing in range(num_things):
        thing = hash_choice([i, i_thing if randomize_things else 0, "thing"], things)
        thing_mask = thing > 0

        if i_thing == 0 or randomize_transform:
            degrees = hash_rand([i, i_thing, "angle"], 0, 360)
            center = (0.5 * thing.shape[1], 0.5 * thing.shape[0])

            # Create rotation matrix
            M = cv2.getRotationMatrix2D(center, degrees, 1)

            # Create random scale matrix with different scale for x and y
            scale_x = hash_rand([i, i_thing, "sx"], 0.7, 1.3)
            scale_y = hash_rand([i, i_thing, "sy"], 0.7, 1.3)
            S = np.array([[scale_x, 0, 0], [0, scale_y, 0]], dtype=np.float32)

            M = mul_affine(S, M)

            # Warp image without cropping
            thing = warp_image_without_cropping(thing, M)

            mask_dilation = hash_randrange([i, i_thing, "mask_dilation"], 1, 30)

            thing = np.pad(thing, mask_dilation)
            thing_mask = cv2.blur(thing, (mask_dilation, mask_dilation))
            thing_mask = thing_mask > 0.01

        if i_thing == 0 or randomize_brightness:
            brightness = hash_rand([i, "brightness"], 0.9, 1.0)

        h, w = thing.shape

        assert image.shape[0] > thing.shape[0]
        assert image.shape[1] > thing.shape[1]

        x = hash_randrange([i, i_thing, "x"], image.shape[1] - w)
        y = hash_randrange([i, i_thing, "y"], image.shape[0] - h)

        assert (h, w) == thing.shape
        assert x + w <= image.shape[1]
        assert y + h <= image.shape[0]
        assert disallowed_mask.shape == image.shape

        if np.sum(thing_mask * disallowed_mask[y:y+h, x:x+w]) == 0:
            image[y:y+h, x:x+w] *= (1 - brightness * thing)
            disallowed_mask[y:y+h, x:x+w] += thing_mask

    return image

def main():
    width = 512
    height = 512
    num_images = 60 if len(sys.argv) < 2 else int(sys.argv[1])
    print(f"Generating {num_images} images")
    lock = threading.Lock()
    counter = 0
    start_time = time.perf_counter()

    # If your computer does not have enough RAM, it should be possible to
    # work around it with mmap (not implemented).
    images = np.zeros((num_images, height, width), dtype=np.uint8)

    things = load_things()

    def generate_image_quantized(i):
        image = generate_image(i + 40_000, width, height, things)

        # Use less memory with uint8 instead of float
        image = np.clip(image * 255, 0, 255).astype(np.uint8)

        images[i] = image

        # Print progress every 100 images
        with lock:
            nonlocal counter
            counter += 1
            if counter % 100 == 0 or counter == num_images:
                elapsed_time = time.perf_counter() - start_time
                remaining_time = (num_images - counter) * elapsed_time / counter
                print(f"{counter:6d}/{num_images}, {remaining_time:.3f} seconds remaining")

    t = time.perf_counter()

    with ThreadPool() as pool:
        pool.map(generate_image_quantized, range(num_images))

    dt = time.perf_counter() - t
    print(dt, "seconds total,", dt / num_images, "seconds per image")

    np.save(images_will_be_written_here, images)

def show_images():
    nx = 4
    ny = 4
    images = np.load(images_will_be_written_here, mmap_mode="r")
    images = images[:ny*nx]
    images = images.reshape(ny, nx, images.shape[1], images.shape[2])
    images = np.concatenate(images, axis=-1)
    images = np.concatenate(images, axis=-2)
    images = cv2.resize(images, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    preview_path = "preview.png"
    cv2.imwrite(preview_path, images)
    print(f"Preview of generated images written to {preview_path}")
    
def convert_numpys_to_pngs(np_path, png_dir):
    os.makedirs(png_dir, exist_ok=True)
    images = np.load(np_path)
    for i, image in enumerate(images):
        cv2.imwrite(os.path.join(png_dir, f"{i}.png"), image)
        

if __name__ == '__main__':
    images_will_be_written_here = "images.npy"
    main()
    show_images()
    convert_numpys_to_pngs(images_will_be_written_here, "generated_data")