import numpy as np
import random
import math
import cv2

class Point:
    def __init__(self, x, y, z, i, j):
        self.i = i
        self.j = j
        self.x = x
        self.y = y
        self.z = z
        self.cluster = -1
        self.centroid = []
        self.min_distance = np.inf

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"

def morfology_filter(image):
    
    rows, columns = image.shape
    inverted = np.zeros(image.shape, dtype=np.uint8)
    result = np.zeros(image.shape, dtype=np.uint8)

    for i in range(rows):
        for j in range(columns):
            if image[i][j] == 0:
                inverted[i][j] = 255
            else:
                inverted[i][j] = 0

    cv2.floodFill(inverted, None, (0, 0), 255)

    for i in range(rows):
        for j in range(columns):
            if image[i][j] != inverted[i][j]:
                result[i][j] = image[i][j] + inverted[i][j]

    return result

def binarize(image, threashold):
    rows, columns, _ = image.shape
    result = np.zeros((rows, columns), dtype=np.uint8)

    for i in range(rows):
        for j in range(columns):

            blue = image[i][j][0]
            green = image[i][j][1]
            red = image[i][j][2]

            if blue >= threashold and green >= threashold and red >= threashold:
                result[i][j] = 255
            elif red >= threashold:
                result[i][j] = 0
            else:
                result[i][j] = 255

    return result

def euclidean_distance(point1, point2):
    x = point1.x - point2.x
    y = point1.y - point2.y
    z = point1.z - point2.z

    return math.sqrt(x**2 + y**2 + z**2)

def kmeans(image, epochs, clusters):
    k = clusters
    centroids = {}
    rows, cols, _ = image.shape

    centroids[0] = Point(0, 0, 0,0, 0)
    centroids[1] = Point(255, 255, 255, 0, 0)
    centroids[2] = Point(255, 0, 0, 0, 0)
    centroids[3] = Point(0, 255, 0, 0, 0)
    centroids[4] = Point(0, 0, 255, 0, 0)
        
    points = []    
    for i in range(rows):
        for j in range(cols):
            blue = image[i][j][0]
            green = image[i][j][1]
            red = image[i][j][2]
            points.append(Point(blue, green, red, i, j))

    for epoch in range(epochs):
        print(f"Etapa {epoch+1}")
        # Assign each point to a cluster
        for i in range(k):
            print(f"- Centroide {i}")
            centroid = centroids[i]
            for point in points:
                distance = euclidean_distance(centroid, point)
                if distance < point.min_distance:
                    point.min_distance = distance
                    point.cluster = i
        
        # Calculate new centroids
        total_centroids_points = np.ones(k, dtype=int)
        sum_centroids_x = np.ones(k, dtype=int)
        sum_centroids_y = np.ones(k, dtype=int)
        sum_centroids_z = np.ones(k, dtype=int)

        for point in points:
            cluster = point.cluster
            total_centroids_points[cluster] += 1
            sum_centroids_x[cluster] += point.x
            sum_centroids_y[cluster] += point.y
            sum_centroids_z[cluster] += point.z

        # Assign new centroids
        for i in range(k):
            x = int(sum_centroids_x[i] / total_centroids_points[i])
            y = int(sum_centroids_y[i] / total_centroids_points[i])
            z = int(sum_centroids_z[i] / total_centroids_points[i])
            centroids[i] = Point(x, y, z, 0 ,0)

    for point in points:
        cluster = point.cluster
        centroid = centroids[cluster]

        red = int(centroid.x)
        green = int(centroid.y)
        blue = int(centroid.z)

        point.centroid = [red, green, blue]

    return points

def tag_image(image):

    rows, cols = image.shape
    image_bordered = np.zeros((rows+2, cols+2), dtype=np.uint8)
    counter = 20
    image_bordered[1:rows+1, 1:cols+1] = image

    for i in range(rows):
        for j in range(cols):
            pixel = image_bordered[i+1][j+1]
            is_noise = False
            for x in range(-1, 2):
                for y in range(-1, 2):
                    if image_bordered[i+x+1][j+y+1] == 0:
                        is_noise = True

            if pixel == 255 and not is_noise:
                cv2.floodFill(image, None, (j, i), 255-counter)
                cv2.floodFill(image_bordered, None, (j+1, i+1), 255-counter)
                counter += 20
    
    groups = {}
    for i in range(rows):
        for j in range(cols):
            pixel = image[i][j]
            if pixel != 0 and pixel != 255:
                if pixel not in groups:
                    groups[pixel] = [(i, j)]
                else:
                    groups[pixel].append((i, j))

    groups_removed = {}
    for key in list(groups.keys()):
        if len(groups[key]) > 100000:
            groups_removed[key] = groups[key]

    pixels = []
    for key in list(groups_removed.keys()):
        pixels.append(groups_removed[key])

    return pixels

def calc_edges(image, group_pixels):
    
    image_result = np.zeros(image.shape, dtype=np.uint8)
    new_groups = []
    for pixels in group_pixels:
        aux = []
        for pixel in pixels:
            i, j = pixel
            is_border = False
            for x in range(-1, 2):
                for y in range(-1, 2):
                    if image[i+x][j+y] == 0:
                        is_border = True
            if is_border:
                aux.append(pixel)
        new_groups.append(aux)
    
    for pixels in new_groups:
        for point in pixels:
            i, j = point
            image_result[i][j] = 255

    return new_groups, image_result

def calc_distances(groups):

    result = []
    for group in groups:
        max_distance = -10000
        max_point = ((0, 0), (0, 0)) 
        for pixel_1 in group:
            x_1, y_1 = pixel_1
            for pixel_2 in group:
                x_2, y_2 = pixel_2
                distance = math.sqrt((x_1 - x_2)**2 + (y_1 - y_2)**2)
                if distance > max_distance:
                    max_distance = distance
                    max_point = ((pixel_1), (pixel_2))
        result.append(max_point)

    return result

def main():
    image = cv2.imread("Jit1_small.jpg", cv2.IMREAD_COLOR)
    print(image.shape)

    clusters = 5
    epoch = 5

    print("Segmentando ...")
    points = kmeans(image, epoch, clusters)
    image_separed = np.zeros(image.shape, dtype=np.uint8)

    for point in points:
        i = point.i
        j = point.j
        image_separed[i][j] = point.centroid
    
    print("Binarizando ...")
    image_binarized = binarize(image_separed, 120)

    print("Eliminando ruido ...")
    image_morfold = morfology_filter(image_binarized)

    print("Etiquetando imagen ...")
    image_tagged = image_morfold.copy()
    pixels = tag_image(image_tagged)

    print("Calculando bordes ...")
    pixels, image_tagged = calc_edges(image_morfold, pixels)
    image_lines = image.copy()

    print("Calculando puntos ...")
    points = calc_distances(pixels)
    print(points)
    image_lines_points = []
    
    print("Imprimiendo imagenes ...")
    for i in range(len(points)):
        max_points = points[i]
        if i % 2 == 1:
            image_lines_points.append(max_points)
            point_1, point_2 = max_points
            x_1, y_1 = point_1
            x_2, y_2 = point_2
            cv2.line(image_lines, (y_1, x_1), (y_2, x_2), (0,255,0), 5)

    point_1_line_1, point_2_line_1 = image_lines_points[0]
    point_1_line_2, point_2_line_2 = image_lines_points[1]

    line_1_length = math.sqrt((point_1_line_1[0] - point_2_line_1[0])**2 + (point_1_line_1[1] - point_2_line_1[1])**2)
    line_2_length = math.sqrt((point_1_line_2[0] - point_2_line_2[0])**2 + (point_1_line_2[1] - point_2_line_2[1])**2)

    print("- Primera linea (mas a la derecha)")
    print(f"Punto 1: ({point_1_line_1[0]}, {point_1_line_1[1]})")
    print(f"Punto 2: ({point_2_line_1[0]}, {point_2_line_1[1]})")
    print(f"La longitud de la recta es: {int(line_1_length)} pixeles")
    print()

    print("- Segunda linea (linea de hasta abajo)")
    print(f"Punto 1: ({point_1_line_2[0]}, {point_1_line_2[1]})")
    print(f"Punto 2: ({point_2_line_2[0]}, {point_2_line_2[1]})")
    print(f"La longitud de la recta es: {int(line_2_length)} pixeles")
    print()

    cv2.imshow("Original image", image)
    cv2.imshow("Image objects", image_separed)
    cv2.imshow("Binarized image", image_binarized)
    cv2.imshow("Image morfologied", image_morfold)
    cv2.imshow("Image tagged", image_tagged)
    cv2.imshow("Final image", image_lines)

    cv2.imwrite("image.jpg", image)
    cv2.imwrite("image_binarized.jpg", image_binarized)
    cv2.imwrite("image_separed.jpg", image_separed)
    cv2.imwrite("image_morfold.jpg", image_morfold)
    cv2.imwrite("image_tagged.jpg", image_tagged)
    cv2.imwrite("image_lines.jpg", image_lines)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()