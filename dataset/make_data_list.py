import numpy as np
import os

cirrus_list = []
spectralis_list = []
topcon_list = []
# topcon2_list = []

path = "D:\\Projects\\Data\\2017FluidChallenge\\FluidSegDataset\\test\\images"

image_list = os.listdir(path)
image_len = len(image_list)

for i in range(image_len):
    image_name = image_list[i]
    name_sp = image_name.split('_')

    if name_sp[0] == "cirrus":
        cirrus_list.append(image_name)
    if name_sp[0] == "spectralis":
        spectralis_list.append(image_name)
    if name_sp[0] == "topcon":
        topcon_list.append(image_name)
    # if name_sp[0] == "topcon2":
    #     topcon2_list.append(image_name)

print(len(cirrus_list))
print(len(spectralis_list))
print(len(topcon_list))
# print(len(topcon2_list))

np.savez('data_list_test.npz', cirrus_list=cirrus_list, spectralis_list=spectralis_list, topcon_list=topcon_list)
