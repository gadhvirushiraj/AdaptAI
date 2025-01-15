import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import librosa

def transcribe_audio(wav_file: str) -> str:
    """
    Transcribes the speech from the provided WAV file.
    
    Args:
        wav_file (str): Path to the WAV file.
    
    Returns:
        str: Transcribed text from the audio file.
    """
    # Define device and model parameters
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Load model and processor
    model_id = "distil-whisper/distil-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        torch_dtype=torch_dtype,
        device=device,
    )

    # Load the audio file
    audio_data, sample_rate = librosa.load(wav_file, sr=16000)  # Ensure it's resampled to 16 kHz

    # Convert audio to the format the pipeline expects
    audio_input = {"array": audio_data, "sampling_rate": sample_rate}

    # Perform transcription
    result = pipe(audio_input)
    return result["text"]
