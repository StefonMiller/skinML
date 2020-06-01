from PIL import Image, ImageFilter
import requests
from io import BytesIO
import skimage.io
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def visualize_colors(cluster, centroids):
    # Get the number of different clusters, create histogram, and normalize
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins = labels)
    hist = hist.astype("float")
    hist /= hist.sum()

    # Create frequency rect and iterate through each cluster's color and percentage
    rect = np.zeros((50, 300, 3), dtype=np.uint8)
    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)])
    start = 0
    for (percent, color) in colors[:-1]:
        print(color, "{:0.2f}%".format(percent * 100))
        end = start + (percent * 300)
        cv2.rectangle(rect, (int(start), 0), (int(end), 50), \
                      color.astype("uint8").tolist(), -1)
        start = end
    return rect

img_response = requests.get('https://steamcommunity-a.akamaihd.net/economy/image/6TMcQ7eX6E0EZl2byXi7vaVKyDk_zQLX05x6eLCFM9neAckxGDf7qU2e2gu64OnAeQ7835BZ42LDfDY0jhyo8DEiv5dRO6g7qrIzRfEzj-qDtw/360fx360f', stream=True).raw

# Once we get the image, load it into a np array
#img = skimage.io.imread(BytesIO(img_response.content))
# Convert low-intensity image colors to 0(black) and convert the image to a PIL object
#img[img < 16] = 0
image = np.asarray(bytearray(img_response.read()), dtype="uint8")
image = cv2.imdecode(image, cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
reshape = image.reshape((image.shape[0] * image.shape[1], 3))
reshape[reshape < 16] = 0

# Find and display most dominant colors
cluster = KMeans(n_clusters=5).fit(reshape)

# TODO: Return pixel data from method, finalize image data output for csv

visualize = visualize_colors(cluster, cluster.cluster_centers_)
visualize = cv2.cvtColor(visualize, cv2.COLOR_RGB2BGR)
cv2.imshow('visualize', visualize)
cv2.waitKey()


'''
img = Image.fromarray(img)
img = img.convert('RGB')

# Get all pixel colors using img.colors. 0, 0, 0 is the last item in the returned array, so to omit the background
# and any low intensity pixels we simply take all items except the last one
img_colors = img.getcolors(360*360)[:-1]

# Sort the array of pixels and print the 10 RGB values with the highest frequency.
# TODO Ensure unique color values!
unique_array = []
sorted_pixels = sorted(img_colors, key=lambda t: t[0], reverse=True)
i = 0
while len(unique_array) <= 9:
    dominant_color = sorted_pixels[i][1]
    if not unique_array:
        unique_array.append(dominant_color)
    else:
        break_flag = False
        for unique in unique_array:
            color1 = convert_color(sRGBColor(unique[0], unique[1], unique[2]), LabColor)
            color2 = convert_color(sRGBColor(dominant_color[0], dominant_color[1], dominant_color[2]), LabColor)
            delta_e = delta_e_cie2000(color1, color2)
            print('Dist is ' + str(delta_e))
            if delta_e < 50:
                break_flag = True
                break

        if break_flag:
            print('\tSkipping...')
        else:
            unique_array.append(dominant_color)
    i += 1

print(unique_array)'''


