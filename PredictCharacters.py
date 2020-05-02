import pickle

from SegmentCharacters import CharacterSegmentator


class CharacterPredicter:

    @staticmethod
    def predict(path, model_path):

        print("===========================================")
        print("--------------> start for %s" % path)
        try:
            model = pickle.load(open(model_path, 'rb'))
        except FileNotFoundError:
            print("ERROR: Wrong path to model (%s)" % model_path)
            return

        segmentor = CharacterSegmentator()
        characters_list, columns_list = segmentor.segment_chars(path)

        counter = 0
        result_count = 0
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
                print("--------------#%d" % (counter + 1))
                print('Predicted license plate : %s' % plate_string)
                print('License plate : %s' % rightplate_string)
                result_count += 1
            counter += 1

        if result_count == 0 :
            print('WARNING: Can not recognize characters in plate')

