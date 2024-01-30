"""
import numpy as np
import pyaudio
from matplotlib import pyplot as plt
import pandas as pd
import sounddevice as sd

from keras_yamnet import params
from keras_yamnet.yamnet import YAMNet, class_names
from keras_yamnet.preprocessing import preprocess_input

from plot import Plotter

if __name__ == "__main__":

    ################### SETTINGS ###################
    plt_classes = [0,132,420,494] # Speech, Music, Explosion, Silence 
    class_labels=True
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = params.SAMPLE_RATE
    WIN_SIZE_SEC = 0.975
    CHUNK = int(WIN_SIZE_SEC * RATE)
    RECORD_SECONDS = 500

    print(sd.query_devices())
    MIC = None

    #################### MODEL #####################
    
    model = YAMNet(weights='keras_yamnet/yamnet.h5')
    yamnet_classes = class_names('keras_yamnet/yamnet_class_map.csv')

    #################### STREAM ####################
    audio = pyaudio.PyAudio()

    # start Recording
    stream = audio.open(format=FORMAT,
                        input_device_index=MIC,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    print("recording...")

    if plt_classes is not None:
        plt_classes_lab = yamnet_classes[plt_classes]
        n_classes = len(plt_classes)
    else:
        plt_classes = [k for k in range(len(yamnet_classes))]
        plt_classes_lab = yamnet_classes if class_labels else None
        n_classes = len(yamnet_classes)

    monitor = Plotter(n_classes=n_classes, FIG_SIZE=(12,6), msd_labels=plt_classes_lab)

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        # Waveform
        data = preprocess_input(np.fromstring(
            stream.read(CHUNK), dtype=np.float32), RATE)
        prediction = model.predict(np.expand_dims(data,0))[0]

        monitor(data.transpose(), np.expand_dims(prediction[plt_classes],-1))

    print("finished recording")

    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
"""
import numpy as np
from matplotlib import pyplot as plt
from keras_yamnet import params
from keras_yamnet.yamnet import YAMNet, class_names
from keras_yamnet.preprocessing import preprocess_input
from plot import Plotter

# Set the path to your static audio file
audio_file_path = 'demo_file.wav'

################### SETTINGS ###################
plt_classes = [0, 132, 420, 494]  # Speech, Music, Explosion, Silence
class_labels = True
FORMAT = np.float32
CHANNELS = 1
RATE = params.SAMPLE_RATE
WIN_SIZE_SEC = 0.975
CHUNK = int(WIN_SIZE_SEC * RATE)

#################### MODEL #####################
model = YAMNet(weights='keras_yamnet/yamnet.h5')
yamnet_classes = class_names('keras_yamnet/yamnet_class_map.csv')

#################### PROCESS AUDIO FILE ####################
audio_data, _ = librosa.load(audio_file_path, sr=RATE, mono=True)
n_frames = len(audio_data) // CHUNK

if plt_classes is not None:
    plt_classes_lab = yamnet_classes[plt_classes]
    n_classes = len(plt_classes)
else:
    plt_classes = [k for k in range(len(yamnet_classes))]
    plt_classes_lab = yamnet_classes if class_labels else None
    n_classes = len(yamnet_classes)

monitor = Plotter(n_classes=n_classes, FIG_SIZE=(12, 6), msd_labels=plt_classes_lab)

for i in range(n_frames):
    start = i * CHUNK
    end = (i + 1) * CHUNK
    data = preprocess_input(audio_data[start:end], RATE)
    prediction = model.predict(np.expand_dims(data, 0))[0]
    monitor(data.transpose(), np.expand_dims(prediction[plt_classes], -1))

print("finished processing audio file")
