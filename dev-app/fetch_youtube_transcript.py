#!/usr/bin/env python3
"""
Fetch YouTube video transcript
"""
from youtube_transcript_api import YouTubeTranscriptApi
import sys

def fetch_transcript(video_id):
    """Fetch transcript for a YouTube video"""
    try:
        # Get the transcript
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])

        # Combine all text
        full_transcript = " ".join([entry['text'] for entry in transcript_list])

        return full_transcript
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return None

if __name__ == "__main__":
    # Extract video ID from URL: https://www.youtube.com/watch?v=hy9Nce__k78
    video_id = "hy9Nce__k78"

    print(f"Fetching transcript for video ID: {video_id}")
    print("=" * 80)

    transcript = fetch_transcript(video_id)

    if transcript:
        print(transcript)
        print("\n" + "=" * 80)
        print(f"Transcript length: {len(transcript)} characters")
    else:
        print("Failed to fetch transcript")
        sys.exit(1)
