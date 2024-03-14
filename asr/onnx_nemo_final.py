#Helper_functions.

#This script allows us to run inferences on audio files using both the Nemo and the ONNX models.


def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum(axis=-1).reshape([logits.shape[0], 1])

def get_nemo_dataset(config, vocab, sample_rate=16000):
    augmentor = None

    config = {
        'manifest_filepath': os.path.join(config['temp_dir'], 'manifest.json'),
        'sample_rate': sample_rate,
        'labels': vocab,
        'batch_size': min(config['batch_size'], len(config['paths2audio_files'])),
        'trim_silence': True,
        'shuffle': False,
    }

    dataset = AudioToCharDataset(
        manifest_filepath=config['manifest_filepath'],
        labels=config['labels'],
        sample_rate=config['sample_rate'],
        int_values=config.get('int_values', False),
        augmentor=augmentor,
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        max_utts=config.get('max_utts', 0),
        blank_index=config.get('blank_index', -1),
        unk_index=config.get('unk_index', -1),
        normalize=config.get('normalize_transcripts', False),
        trim=config.get('trim_silence', True),
        #load_audio=config.get('load_audio', True),
        parser=config.get('parser', 'en'),
        #add_misc=config.get('add_misc', False),
    )

    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config['batch_size'],
        collate_fn=dataset.collate_fn,
        drop_last=config.get('drop_last', False),
        shuffle=True,
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', False),
    )

def get_letters(probs):
    letters = []
    for idx in range(0, probs.shape[0]):
        current_char_idx = np.argmax(probs[idx])
        if labels[current_char_idx] != "blank":
            letters.append([labels[current_char_idx], idx])
    return letters

#__________________________________________________________________________________________________________________________________________


audio_files_set = [
"/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_02931.wav",
"/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_02773.wav",
"/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_03265.wav",
"/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_01628.wav",
"/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_00706.wav",
 "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_02206.wav",
 "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_04370.wav",
 "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_00756.wav",
 "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_02223.wav",
 "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_03338.wav",
 "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_00660.wav",
 "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_03532.wav",
 "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_02136.wav",
 "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_01817.wav",
 "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_04072.wav",
 "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_04582.wav",
 "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_03324.wav",
 "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_02415.wav",
 "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_01208.wav",
 "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_01582.wav", 
  "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_04280.wav", 
  "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_03545.wav",
  "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_03623.wav",
  "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_02898.wav",
  "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_04468.wav",
  "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_03256.wav",
  "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_00539.wav",
  "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_01788.wav",
  "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_03297.wav", 
  "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_02733.wav",
  "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_02608.wav",
  "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_02724.wav",
  "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_00650.wav",
  "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_04277.wav",
  "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_02851.wav",
  "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_02293.wav",
  "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_01638.wav", 
  "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_02439.wav",
  "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_01959.wav",
  "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_00798.wav", 
  "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_02497.wav",
  "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_03115.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_00100.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_00152.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_00803.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_03123.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_01872.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_02688.wav", 
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_01294.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_04010.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_00715.wav", 
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_03469.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_03719.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_03168.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_01702.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_02942.wav", 
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_00757.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_01185.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_00808.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_02278.wav", 
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_03259.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_01654.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_02960.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_01259.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_03220.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_00538.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_02177.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_00424.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_00272.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_03216.wav", 
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_01501.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_03243.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_02047.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_01543.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_02345.wav", 
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_02164.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_02746.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_00384.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_01212.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_02894.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_02009.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_01642.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_01034.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_01720.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_01839.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_00600.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_00545.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_01919.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_02062.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_00674.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_04365.wav", 
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_02600.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_04583.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_04374.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_01957.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_00814.wav", 
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_03580.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_04354.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_03561.wav",
   "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullmale_02282.wav",  


 
]

#__________________________________________________________________________________________________________________________________________


#Onnx Evaluation.
import onnxruntime
import tempfile
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.data.audio_to_text import AudioToCharDataset
import os
import torch
import yaml
from omegaconf import DictConfig
import json
import numpy as np

# your original  model config
config_path = "/mnt/sangraha/venkat_shadeslayer/asr/momodels/stt_hi_conformer_ctc_medium.yaml"
# your exported onnx model path
model_to_load = "/mnt/sangraha/venkat_shadeslayer/asr/momodels/mymodel3.onnx"

with open(config_path) as f:
    params = yaml.safe_load(f)

# create onnx session with model (assuming you already exported your model)
sess = onnxruntime.InferenceSession(model_to_load)
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

# create preprocessor (NeMo does this inside EncDecCTCModel, here we do it explicitly since we are going to use onnx and not EncDecCTCModel)
preprocessor_cfg = DictConfig(params).preprocessor
preprocessor = EncDecCTCModel.from_config_dict(preprocessor_cfg)

audio_file = "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_00803.wav"
labels = params['decoder']['vocabulary'] #"vocabulary" could be something else depending on model

# this part is just copy pasted from NeMo library code
with tempfile.TemporaryDirectory() as dataloader_tmpdir:
    with open(os.path.join(dataloader_tmpdir, 'manifest.json'), 'w') as fp:
        entry = {'audio_filepath': audio_file, 'duration': 1000, 'text': 'nothing'}
        fp.write(json.dumps(entry) + '\n')
    out_batch = []
    config = {'paths2audio_files': [audio_file], 'batch_size': 1, 'temp_dir': dataloader_tmpdir}
    temporary_datalayer = get_nemo_dataset(config, labels, 16000)
    for test_batch in temporary_datalayer:
        out_batch.append(test_batch)

# preprocess audio just like NeMo does
processed_signal, processed_signal_length = preprocessor(input_signal=out_batch[0][0], length=out_batch[0][1], )
processed_signal = processed_signal.cpu().numpy()

# finally make an inference using onnx model
logits = sess.run([],{input_name:processed_signal,'length':processed_signal_length.cpu().numpy()})
probabilities = logits[0][0]

probs = softmax(probabilities)


#silence
labels.append("blank")

letters = get_letters(probs)

print(letters)


#Uptil here we get the outputs just like how the ctc model gives(Repeated output tokens and  the blank token.)
#Now we pass the probab outputs through a ctc decoder and get the text in a clear sentence format.


from pyctcdecode import build_ctcdecoder 
import nemo.collections.asr as nemo_asr

asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
  model_name='stt_hi_conformer_ctc_medium'
)
#logits = asr_model.transcribe(["/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_00803.wav"], logprobs=True)[0]

decoder_inputs = np.array(probs)



decoder = build_ctcdecoder(asr_model.decoder.vocabulary)

output = decoder.decode(decoder_inputs)
print(f"onnx_output:",output)


#__________________________________________________________________________________________________________________________________________


#NeMo Evaluation

import onnx
import torch
onnx_model = onnx.load("/mnt/sangraha/venkat_shadeslayer/asr/momodels/mymodel2.onnx")
onnx.checker.check_model(onnx_model)


import onnxruntime as ort
import numpy as np
import librosa

ort_sess = ort.InferenceSession('/mnt/sangraha/venkat_shadeslayer/asr/momodels/mymodel3.onnx')

audio_filepath = "/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_00803.wav"
audio_signal, sr = librosa.load(audio_filepath, sr=16000)


length_data = 16000
# audio_signal = np.random.random((1,80,length_data))
# audio_signal = np.array(audio_signal,dtype=np.float32)
S = librosa.feature.melspectrogram(y=audio_signal, sr=sr, n_mels=80)
print(S.shape)
S_new = np.reshape(S, (1, 80, -1))
print(S_new.shape)

len(audio_signal)


outputs = ort_sess.run(None, {'audio_signal': S_new, 'length': np.array([length_data],dtype=np.int64)})
outputs[0].squeeze(0)


#Helper functions
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

original_input = outputs

softmax_outputs = softmax(original_input)

print("Original array:", original_input)
print("Softmax result:", softmax_outputs)


from pyctcdecode import build_ctcdecoder 
import nemo.collections.asr as nemo_asr


asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
  model_name='stt_hi_conformer_ctc_medium'
)
logits = asr_model.transcribe(["/mnt/sangraha/venkat_shadeslayer/asr/indictts/hindi/wavs/train_hindifullfemale_00803.wav"], logprobs=True)[0]

decoder_inputs = np.array(original_input)



decoder = build_ctcdecoder(asr_model.decoder.vocabulary)

output = decoder.decode(logits)
print(f"Nemo_output:",output)

