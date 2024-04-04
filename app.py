import torch
from TTS.api import TTS

# check for gpu
if torch.cuda.is_available():
    device = "cuda"
    print("\nUsing GPU\n") 
else:
    device = "cpu"
    print("\nNo GPU found, using CPU only\n") 

# initialise TTS
mySpeech = "Obtained my Diploma of Information Technology from Curtin College in 2021 (AQF Certificate level V), I am currently in my last semester of BSc.Computer Science at the University of London (remote) specializing in Artificial Intelligence (AI) and Machine Learning (ML). With a solid academic foundation in programming, problem identification, and problem-solving, I am self-motivated and efficiency-driven. I have also demonstrated the ability to learn and adapt to new technologies and excel both independently and as part of a collaborative environment. I am very keen on leveraging AI and ML techniques to contribute to national and organizational success and growthI am eager to embark on my career journey and apply my skills and knowledge to make meaningful contributions to the industry."
#tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False).to("cuda")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to("cuda")
tts.tts_to_file(mySpeech, speaker_wav="./data/naufal_short.wav", language="en", file_path="bio_V2.wav")