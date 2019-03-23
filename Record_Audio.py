import pyaudio
import wave

CHUNK = 1024 
FORMAT = pyaudio.paInt16 #paInt8
CHANNELS = 2 
RATE = 44100 #sample rate
RECORD_SECONDS = 4
WAVE_OUTPUT_FILENAME = "output12.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK) #buffer

print("* recording")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data) # 2 bytes(16 bits) per channel

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
'''
from deepaffects.realtime.util import chunk_generator_from_file, chunk_generator_from_url, get_deepaffects_client

TIMEOUT_SECONDS = 3000
apikey = 'ETbUqPSIATF2z2KOdodACgMKh78dhTKM'

# Set file_path as local file path or audio stream or youtube url
file_path = "output10.wav"

# Set is_youtube_url True while streaming from youtube url
is_youtube_url = False
languageCode = "en-Us"
sampleRate = "16000"
encoding = "wav"

# DeepAffects realtime Api client
client = get_deepaffects_client()

metadata = [
    ('apikey', apikey),
    ('encoding', encoding),
    ('samplerate', sampleRate),
    ('languagecode', languageCode)
]

"""Generator Function

chunk_generator_from_file is the Sample implementation for generator funcion which reads audio from a file and splits it into
base64 encoded audio segment of more than 3 sec
and yields SegmentChunk object using segment_chunk

"""

# from deepaffects.realtime.types import segment_chunk
# segment_chunk(Args)

"""segment_chunk.

Args:
    encoding : Audio Encoding,
    languageCode: language code ,
    sampleRate: sample rate of audio ,
    content: base64 encoded audio,
    segmentOffset: offset of the segment in complete audio stream
"""

# Call client api function with generator and metadata

responses = client.IdentifyEmotion(
    # Use chunk_generator_from_file generator to stream from local file
    chunk_generator_from_file(file_path),
    # Use chunk_generator_from_url generator to stream from remote url or youtube with is_youtube_url set to true
    # chunk_generator_from_url(file_path, is_youtube_url=is_youtube_url),
     TIMEOUT_SECONDS, metadata=metadata)

# responses is the iterator for all the response values
for response in responses:
    print("Received message",response)
    break

"""Response.
    response = {
        emotion: Emotion identified in the segment,
        start: start of the segment,
        end: end of the segment
    }
"""
'''

