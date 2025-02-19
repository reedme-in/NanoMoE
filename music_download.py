import os
import subprocess
from pathlib import Path
from datasets import load_dataset, Audio

def download_clip(ytid, start_time, end_time, output_filename, num_attempts=5):
    """
    Download a 10-second clip from a YouTube video using yt-dlp.
    
    Args:
        ytid (str): YouTube video ID.
        start_time (int or float): Start time (in seconds) for the clip.
        end_time (int or float): End time (in seconds) for the clip.
        output_filename (str): Path to save the downloaded file.
        num_attempts (int): Number of attempts to retry downloading.
    
    Returns:
        bool: True if download succeeded, False otherwise.
    """
    # Construct the yt-dlp command. The "--download-sections" option downloads only the segment.
    command = (
        f'yt-dlp --quiet --no-warnings -x --audio-format wav '
        f'-f bestaudio -o "{output_filename}" --download-sections "*{start_time}-{end_time}" '
        f"https://www.youtube.com/watch?v={ytid}"
    )
    
    for attempt in range(num_attempts):
        try:
            subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
            if os.path.exists(output_filename):
                return True
        except subprocess.CalledProcessError as e:
            print(f"Attempt {attempt+1} failed for video {ytid}: {e.output.decode()}")
    return False

def download_musiccaps(data_dir, limit=None, sampling_rate=44100, num_proc=1):
    """
    Download audio clips from the MusicCaps dataset.
    
    Args:
        data_dir (str): Directory to store downloaded audio files.
        limit (int, optional): Limit the number of examples to process.
        sampling_rate (int): Sampling rate for the downloaded audio.
        num_proc (int): Number of processes to use for parallel downloading.
    
    Returns:
        Dataset: Updated MusicCaps dataset with local audio file paths.
    """
    # Load the MusicCaps train split from Hugging Face.
    ds = load_dataset("google/MusicCaps", split="train")
    if limit is not None:
        ds = ds.select(range(limit))
    
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)
    
    def process(example):
        # Build an output filename based on the YouTube video ID.
        outfile = str(data_dir / f"{example['ytid']}.wav")
        if not os.path.exists(outfile):
            status = download_clip(example["ytid"], example["start_s"], example["end_s"], outfile)
        else:
            status = True
        example["audio_filepath"] = outfile
        example["downloaded"] = status
        return example
    
    ds = ds.map(process, num_proc=num_proc)
    # Cast the downloaded file column as an Audio feature to facilitate further processing.
    ds = ds.cast_column("audio_filepath", Audio(sampling_rate=sampling_rate))
    return ds

if __name__ == "__main__":
    # Example usage: download 100 clips using 4 processes and save them in the "musiccaps_audio" directory.
    dataset = download_musiccaps("musiccaps_audio", limit=100, num_proc=4)
    print(dataset)

