import numpy as np
import matplotlib.pyplot as plt
import cv2

import transformation_functions
import render_functions

# setup constants
H = 15
W = 15
rows = 512
columns = 512

# load data from file
data = np.load("h2.npy", allow_pickle=True)
data = data.item()

# import data
verts3d = data["verts3d"]
c_org = data["c_org"]
c_lookat = data["c_lookat"]
c_up = data["c_up"]
focal = data["focal"]
vcolors = data["vcolors"]
faces = data["faces"]

t_1 = data["t_1"]
t_2 = data["t_2"]
u = data["u"]
phi = data["phi"]

# initial image (pre transformation)
img_one = render_functions.render_object(verts3d, faces, vcolors, H, W, rows, columns, focal, c_org, c_lookat, c_up)
img_one = np.transpose(img_one, axes=(1, 0, 2))

# image moved by t1
dummy_u = np.array([1, 1, 1])
origin = np.array([0, 0, 0])
verts3d = transformation_functions.rotate_translate(verts3d, 0, dummy_u, origin, t_1)

img_two = render_functions.render_object(verts3d, faces, vcolors, H, W, rows, columns, focal, c_org, c_lookat, c_up)
img_two = np.transpose(img_two, axes=(1, 0, 2))

# image rotated by phi around an axis parallel to u
zero_t = np.array([0, 0, 0])
verts3d = transformation_functions.rotate_translate(verts3d, phi, u, origin, zero_t)

img_three = render_functions.render_object(verts3d, faces, vcolors, H, W, rows, columns, focal, c_org, c_lookat, c_up)
img_three = np.transpose(img_three, axes=(1, 0, 2))

# image moved by t2
verts3d = transformation_functions.rotate_translate(verts3d, 0, dummy_u, origin, t_2)

img_four = render_functions.render_object(verts3d, faces, vcolors, H, W, rows, columns, focal, c_org, c_lookat, c_up)
img_four = np.transpose(img_four, axes=(1, 0, 2))

# display images
fig, axs = plt.subplots(2, 2)

fig.set_figheight(10)
fig.set_figwidth(16)
fig.suptitle(f"$t_1$ = {t_1}, $\phi$ = {phi}, $u$ = {u}, $t_2$ = {t_2}, $c_v$ = {np.squeeze(c_org)}, $c_K$ = {np.squeeze(c_lookat)}, $c_u$ = {np.squeeze(c_up)}, $f$ = {focal}")

axs[0, 0].imshow(img_one)
axs[0, 0].set_title('Image without any transforms')
axs[0, 1].imshow(img_two)
axs[0, 1].set_title('Moved by $t_1$')
axs[1, 0].imshow(img_three)
axs[1, 0].set_title('Moved by $t_1$ and rotated by $\phi$ around $u$')
axs[1, 1].imshow(img_four)
axs[1, 1].set_title('Moved by $t_1$ and rotated by $\phi$ around $u$ and moved by $t_2$')

plt.savefig("total_plot.png")
plt.show()

# save images
images = [img_one, img_two, img_three, img_four]

for img_idx in range(len(images)):
    img = images[img_idx]
    img = img * 255.0
    img = img.astype("uint8")
    img = img[:, :, ::-1] # this means read the channels (third dimension) backwards
    cv2.imwrite(f"{img_idx}.jpg", img)
