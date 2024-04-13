#Assuming we have the manifest created, this script is going to be for transcribing the text using Nemo model and storing them onto a text file.
#This script allows us to transcribe any number of .wav files and store the text into a text file.

#_______________________________________________________________________________________________


#Helper_functions.

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

#_______________________________________________________________________________________________


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











for i in audio_files_set:
    audio_file = i


    import onnx
    import torch
    onnx_model = onnx.load("/mnt/sangraha/venkat_shadeslayer/asr/momodels/mymodel3.onnx")
    onnx.checker.check_model(onnx_model)


    import onnxruntime as ort
    import numpy as np
    import librosa

    ort_sess = ort.InferenceSession('/mnt/sangraha/venkat_shadeslayer/asr/momodels/mymodel3.onnx')

    audio_filepath = i
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
    logits = asr_model.transcribe([i])[0]

    # Print and store in a text file
    output_path = "/mnt/sangraha/venkat_shadeslayer/asr/outputs/Nemo_Outputs.txt"
    with open(output_path, 'a') as output_file:
        print(f"Audio File: {audio_file}, Transcription: {logits}")
        output_file.write(f"Audio File: {audio_file}, Transcription: {logits}\n")
