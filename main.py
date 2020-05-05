from src.PredictCharacters import CharacterPredicter
from src.plate_detector.detecor import PlateDetector2

if __name__ == "__main__":

    #detector = PlateDetector2()
    #detector.detect('./examples/tayota_1.jpg')

    CharacterPredicter.predict("examples/carsh_12.jpg", './finalized_model4.sav')
