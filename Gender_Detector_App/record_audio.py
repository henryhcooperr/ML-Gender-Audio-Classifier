import sounddevice as sd
from scipy.io.wavfile import write
import os
from pydub import AudioSegment


def record_sample(duration=5, sample_rate=44100, file_name="output.wav"):
    """
    Record audio from the microphone and save it directly to a WAV file.
   
    Args:
        duration (int): Recording duration in seconds.
        sample_rate (int): Sampling rate of the audio.
        file_name (str): Name of the file to save the recording.
    
    Returns:
        str: Full path to the saved audio file.
    """
    print("Recording audio...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")

    # Specify the directory to save audio
    directory = os.path.join(os.getcwd(), "recorded_audio")
    if not os.path.exists(directory):
        os.makedirs(directory)

    full_path = os.path.join(directory, file_name)
    # Save the recording as a WAV file
    write(full_path, sample_rate, recording)
    print(f"Audio saved to {full_path}")
    
    return full_path

def convert_audio_to_wav(input_path):
    audio = AudioSegment.from_file(input_path)
    output_path = input_path.replace('.mp3', '.wav')  # Change extension to .wav
    audio.export(output_path, format='wav')
    return output_path