from datetime import datetime

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from skimage.measure import regionprops
from skimage.transform import resize

from src.plate_detector.detecor import PlateDetector2


class CharacterSegmentator:

    def segment_chars(self, filename):

        plate_detector = PlateDetector2()
        plate_detector.detect(filename)

        characters_list = []
        column_list = []
        image_outputs_list = []

        counter = 0
        for plate in plate_detector.get_found_plates():
            characters = []
            column = []
            # The invert wasdone so as to convert the black pixel to white pixel and vice versa
            license_plate = np.invert(plate)

            labelled_plate = measure.label(license_plate)

            fig, ax1 = plt.subplots(1)
            ax1.imshow(license_plate, cmap="gray")
            # the next two lines is based on the assumptions that the width of
            # a license plate should be between 5% and 15% of the license plate,
            # and height should be between 35% and 60%
            # this will eliminate some
            character_dimensions = (
                0.3 * license_plate.shape[0], 0.90 * license_plate.shape[0], 0.035 * license_plate.shape[1],
                0.14 * license_plate.shape[1])
            min_height, max_height, min_width, max_width = character_dimensions

            for regions in regionprops(labelled_plate):
                y0, x0, y1, x1 = regions.bbox
                region_height = y1 - y0
                region_width = x1 - x0

                if min_height < region_height < max_height and min_width < region_width < max_width:
                    roi = license_plate[y0:y1, x0:x1]

                    # draw a red bordered rectangle over the character.
                    rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red",
                                                    linewidth=2, fill=False)
                    ax1.add_patch(rect_border)

                    # resize the characters to 20X20 and then append each character into the characters list
                    resized_char = resize(roi, (20, 20), mode='constant', anti_aliasing=False)
                    characters.append(resized_char)

                    # this is just to keep track of the arrangement of the characters
                    column.append(x0)
            # print(characters)
            characters_list.append(characters)
            column_list.append(column)
            output_file_name = "output/%s" % datetime.now().strftime("%Y_%m_%d %H_%M_%S")
            if counter != 0:
                output_file_name += "_%d" % counter
            image_outputs_list.append(output_file_name)
            plt.savefig("%s.jpg" % output_file_name, cmap='gray')
            plt.show()
            counter+=1

        return characters_list, column_list, image_outputs_list
