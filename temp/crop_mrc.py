from mrc_utils import load_mrc, save_im_mrc
import mrcfile
import os
import numpy as np


if __name__ == "__main__":
    fpath = r"/home/astar/Projects/vesicles_data/patri_subtractions/reconstructions/subtracted_mrc/FoilHole_14439761_Data_14437927_14437929_20231114_214246_fractions_aligned_mic_DW.mrc"
    im, header, voxel_size = load_mrc(fpath)
    im = im[2999:3999,3671:4786]
    dir_path = r"/home/astar/Projects/vesicles_data/patri_test_crop/subtracted"
    print("Header cella {} ".format(header["cella"]))
    print("Voxel size: {}".format(voxel_size))
    # Update the header cella values based on the new image shape and voxel size
    with mrcfile.new(os.path.join(dir_path,"test_crop.mrc"), overwrite=True) as mrc_new:
        mrc_new.set_data(im)
        mrc_new.voxel_size = np.rec.array(( voxel_size[0],  voxel_size[1], voxel_size[2]), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])

    #save_im_mrc(im, os.path.join(dir_path,"cropped_im.mrc"),header)