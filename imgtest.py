import requests
import cv2
import numpy as np
from sklearn.cluster import KMeans

# Class containing image color data
# r: red pixel value
# g: green pixel value
# b: blue pixel value
# c: number of pixels in the image with this color
# p: percentage of pixels in the image that are this color
class ImgColor:
    def __init__(self, colors, count, percent):
        self.r = colors[0]
        self.g = colors[1]
        self.b = colors[2]
        self.c = count
        self.p = percent


# Loads image located at url and get the dominant colors of that image
# @Param img_response: Response from a request to the image's url
# @Param num_colors: How many clusters to use in KMeans
# @Return: A list of ImgColor objects
def load_image(img_response, num_colors):
    # Get raw response data from url request and convert it to a cv2 image

    img = np.asarray(bytearray(img_response.read()), dtype="uint8")
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    # Rearrange cv2 from bgr to rgb
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Reshape the image to RGB format and onvert any background pixels to 0,0,0 so we can filter them later
    reshape = img.reshape((img.shape[0] * img.shape[1], 3))
    reshape[reshape < 16] = 0

    # Use KNN to get top image colors
    cluster = KMeans(n_clusters=num_colors).fit(reshape)
    top_colors = get_top_colors(cluster, cluster.cluster_centers_)

    #Once we have the top colors, return them
    return top_colors

# Gets the most dominant colors of an image using KMeans
# @Param cluster: Number of colors(clusters)
# @Param centroids: Center of each cluster used to group data
# @Return color_list: List of ImgColor Objects
def get_top_colors(cluster, centroids):
    # Get the number of different clusters, create histogram, and normalize
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins=labels)
    hist = hist.astype("float")
    hist /= hist.sum()

    color_list = []
    # Create list of colors and their respective percentages
    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)])

    # Since we are omitting any completely black pixels, count how many of the we omit and subtract that total from
    # The total number of pixels in the image. Since we are always using 360x360 images we know the total amount is
    # 360*360
    new_pixels = (360 * 360) - (colors[-1][0] * 360 * 360)

    # For every color, calculate the amount of pixels in the image and their relative percentage. Then create an
    # ImgColor object and add it to our color_list
    for (percent, color) in colors[:-1]:
        count = percent * (360 * 360)
        color_list.append(ImgColor(color, count, ((count / new_pixels) * 100)))

    return color_list


# Prevent anything from running when imported
def main():
    pass

if __name__ == "__main__":
    main()
