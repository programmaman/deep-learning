import os
import time

import numpy as np

data_modal_dictionary = {
    "EDA": "EDA",
    "MMHG": "BP",
    "MEAN": "LA Mean BP",
    "SYS": "LA Systolic BP",
    "PULSE": "Pulse Rate",
    "DIA": "BP Dia",
    "VOLT": "Resp",
    "RESP": "Respiration Rate",
    "ALL":  ""
}


def create_dataset(path, physio_data_modal):
    temporal_physio_data_list = []  # Each element in this list is an array of temporal data
    emotion = []  # Classifies the emotion with an integer from 0 to 9
    padding = 0
    iterations = 0

    if physio_data_modal == "ALL":
        for file in os.listdir(path):
            delimited_file_name = file.split('_')
            emotion_classifier = delimited_file_name[1]
            temporal_data_container = np.loadtxt(path + '\\' + file)
            temporal_physio_data_list.append(temporal_data_container)
            emotion.append(int(emotion_classifier) - 1)
            iterations = iterations + 1
            print(file + " loaded.")
        print("Loaded " + str(iterations) + " files.")
        time.sleep(2)
        return temporal_physio_data_list, emotion
    else:
        for file in os.listdir(path):
            delimited_file_name = file.split('_')
            emotion_classifier = delimited_file_name[1]
            if data_modal_dictionary.get(physio_data_modal) == delimited_file_name[2]:
                temporal_data_container = np.loadtxt(path + '\\' + file)
                temporal_physio_data_list.append(temporal_data_container)
                emotion.append(int(emotion_classifier) - 1)
                iterations = iterations + 1
                print(file + " loaded.")
        print("Loaded " + str(iterations) + " files.")
        time.sleep(2)
        return temporal_physio_data_list, emotion




