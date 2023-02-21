import argparse

parser = argparse.ArgumentParser(description='Speech enhancement,data creation, training and prediction')

# 程序运行模式（创建数据、训练、预测）
parser.add_argument('--mode', default='prediction', type=str, choices=['data_creation', 'training', 'prediction'])
# 存放噪音数据和干净语音数据的文件夹(数据创建模式)
parser.add_argument('--noise_dir', default='/home/ron/SpeechEnhancement/Train/Noise/', type=str)

parser.add_argument('--voice_dir', default='/home/ron/SpeechEnhancement/Train/CleanVoice/',
                    type=str)
# 存放频谱图、时序图以及训练用声音数据的文件夹
parser.add_argument('--path_save_spectrogram',
                    default='/home/ron/SpeechEnhancement/Train/Spectrogram/', type=str)

parser.add_argument('--path_save_time_serie',
                    default='/home/ron/SpeechEnhancement/Train/TimeSeries/', type=str)

parser.add_argument('--path_save_sound', default='/home/ron/SpeechEnhancement/Train/Sound/',
                    type=str)
# 在数据创建模式下需要创建多少帧
parser.add_argument('--nb_samples', default=100000, type=int)
# 从0开始训练或者预训练权重
parser.add_argument('--training_from_scratch', default=True, type=bool)
# 保存权重的文件夹
parser.add_argument('--weights_folder', default='./weights', type=str)
# 训练多少轮
parser.add_argument('--epochs', default=50, type=int)
# 训练的批次大小
parser.add_argument('--batch_size', default=64, type=int)
# 要读取的已保存的模型名称
parser.add_argument('--name_model', default='model_unet', type=str)
# 带噪语音存放位置（预测模式)
parser.add_argument('--audio_dir_prediction', default='./demo_data/test', type=str)
# 去噪语音存放位置（预测模式）
parser.add_argument('--dir_save_prediction', default='./demo_data/save_predictions/', type=str)
# 需要去噪的带噪语音
parser.add_argument('--audio_input_prediction', default=['noisy_voice_long_t2.wav'], type=list)
# 去噪预测声音文件输出的名字
parser.add_argument('--audio_output_prediction', default='denoise_t2.wav', type=str)
# 读取声音文件的采样率
parser.add_argument('--sample_rate', default=16000, type=int)
# 需要考虑的最小语音间隔
parser.add_argument('--min_duration', default=1.0, type=float)
# 训练数据加窗至少一秒以上
parser.add_argument('--frame_length', default=8064, type=int)
# 干净语音分割的跳跃间隔（无重叠）
parser.add_argument('--hop_length_frame', default=8064, type=int)
# 混合噪声的跳跃间隔（噪音被分割为若干个窗）
parser.add_argument('--hop_length_frame_noise', default=5000, type=int)
# 选择n_fft和hop_lenth_fft来得到平均功率谱
parser.add_argument('--n_fft', default=255, type=int)

parser.add_argument('--hop_length_fft', default=63, type=int)
