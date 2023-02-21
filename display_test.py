import librosa
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['FangSong']  # 用来正常显示中文标签,指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示符号，解决负号“-”显示为方块的问题
fs = 16000

wav_data, _ = librosa.load("/home/ron/下载/Speech-enhancement/demo_data/validation/noisy_voice_alarm39.wav", sr=fs, mono=True)

# 画图
plt.subplot(2, 2, 1)
plt.title("Spectrogram", fontsize=15)
plt.specgram(wav_data, Fs=16000, scale_by_freq=True, sides='default', cmap="jet")
plt.xlabel('Time', fontsize=15)
plt.ylabel('Frequency', fontsize=15)

plt.subplot(2, 2, 2)
plt.title("Time Serie", fontsize=15)
time = np.arange(0, len(wav_data)) * (1.0 / fs)
plt.plot(time, wav_data)
plt.xlabel('Time', fontsize=15)
plt.ylabel('Amplitude', fontsize=15)

plt.tight_layout()
plt.show()
