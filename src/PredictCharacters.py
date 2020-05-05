import pickle
from datetime import datetime

from src.SegmentCharacters import CharacterSegmentator


class CharacterPredicter:

    @staticmethod
    def predict(path, model_path):

        print("\n================================================")
        print("========> start for %s" % path)
        print("================================================")
        try:
            model = pickle.load(open(model_path, 'rb'))
        except FileNotFoundError:
            print("ERROR: Wrong path to model (%s)" % model_path)
            return

        segmentor = CharacterSegmentator()
        characters_list, columns_list, files = segmentor.segment_chars(path)

        counter = 0
        results = []
        for characters in characters_list:

            classification_result = []
            for each_character in characters:
                # converts it to a 1D array
                each_character = each_character.reshape(1, -1)
                result = model.predict(each_character)
                classification_result.append(result)

            plate_string = ''
            for eachPredict in classification_result:
                plate_string += eachPredict[0]

            # it's possible the characters are wrongly arranged
            # since that's a possibility, the column_list will be
            # used to sort the letters in the right order

            column_list_copy = columns_list[counter][:]
            columns_list[counter].sort()
            rightplate_string = ''
            for each in columns_list[counter]:
                rightplate_string += plate_string[column_list_copy.index(each)]

            if rightplate_string or plate_string:
                print("------#%d" % (counter + 1))
                print('Predicted license plate : %s' % plate_string)
                print('License plate : %s' % rightplate_string)
                results.append(rightplate_string)

        if len(results) == 0:
            print('WARNING: Can not recognize characters in plate')
            CharacterPredicter.print_to_file(None, files)
        else:
            CharacterPredicter.print_to_file(results, files)

    @staticmethod
    def print_to_file(detected_plate, files=None):
        f = open("log.txt", "a+")

        if detected_plate is not None:
            f.write("%s:  %s -> %s\r\n" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), detected_plate, files))
        else:
            if files is None:
                f.write("%s:  %s\r\n" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Plate Not Detected"))
            else:
                f.write("%s:  Can not recognize characters -> %s\r\n" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), files))

        f.close()
