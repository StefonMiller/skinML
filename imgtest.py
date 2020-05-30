import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from matplotlib import colors
from scipy.spatial import cKDTree as KDTree
import requests
from io import BytesIO

img_response = requests.get('https://steamcommunity-a.akamaihd.net/economy/image/6TMcQ7eX6E0EZl2byXi7vaVKyDk_zQLX05x6eLCFM9neAckxGDf7qU2e2gu64OnAeQ7835FZ4GPEfDY0jhyo8DEiv5dRMak2qbY1R_hiiRlpdA/360fx360f')

# Once we get the image, load it into a np array
img = Image.open(BytesIO(img_response.content))
img = img.convert('RGB')
img_colors = img.getcolors(360*360)[:-100]
print(img_colors)
color_list = []
for color in img_colors:
    for _ in range(color[0]):
        color_list.append(color[1])
all_colors = False

if not all_colors:
        use_colors = {k: colors.cnames[k] for k in ['red', 'green', 'blue', 'black', 'yellow', 'purple']}
else:
    use_colors = colors.cnames

# translate hexstring to RGB tuple
named_colors = {k: tuple(map(int, (v[1:3], v[3:5], v[5:7]), 3 * (16,)))
                for k, v in use_colors.items()}
ncol = len(named_colors)

if not all_colors:
    ncol -= 1
    no_match = named_colors.pop('purple')
else:
    no_match = named_colors['purple']

# make an array containing the RGB values
color_tuples = list(named_colors.values())
color_tuples.append(no_match)
color_tuples = np.array(color_tuples)

color_names = list(named_colors)
color_names.append('no match')

# build tree
tree = KDTree(color_tuples[:-1])
# tolerance for color match `inf` means use best match no matter how
# bad it may be
tolerance = np.inf
# find closest color in tree for each pixel in picture

# Once we get the image, load it into a np array
img = np.array(color_list)

# find closest color in tree for each pixel in picture
dist, idx = tree.query(img, distance_upper_bound=tolerance)

counts = dict(zip(color_names, np.bincount(idx.ravel(), None, ncol + 1)))
count_vals = list(counts.values())
count_vals = [str(i) for i in count_vals]

print(color_names)
print(count_vals)