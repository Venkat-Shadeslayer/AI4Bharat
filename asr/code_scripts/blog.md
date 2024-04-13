STEP1: Convert the .nemo model to .onnx format(The .nemo model is "stt_hi_conformer_ctc_medium.nemo" in this case)

Installation guide in a conda environment.

Please note: All the paths in the commands have been given wrt my localmachine. Please change the paths, the folders and the directories according to your workspace.

Clone the repository in your conda environment by running :  git clone https://github.com/NVIDIA/NeMo.git

Create a directory: asr->code_scripts->nemo(This is the folder from NeMo.nemo)

Required Installations:
pip install torch torchvision torchaudio
pip install hydra-core --upgrade
pip install pytorch_lightning
pip install huggingface-hub
pip install wget
pip install onnx
pip install librosa
pip install transformers
pip install sentencepiece
pip install pandas
pip install inflect
pip install unidecode
pip install lhotse
pip install editdistance
pip install jiwer
pip install pyannote.core pyannote.parser pyannote.database pyannote.metrics
pip install webdataset
pip install datasets
pip install IPython

wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_hi_conformer_ctc_medium/versions/1.6.0/zip -O stt_hi_conformer_ctc_medium_1.6.0.zip

unzip the zipped file using unzip /mnt/sangraha/venkat_shadeslayer/asr/momodels/stt_hi_conformer_ctc_medium_1.6.0.zip

python3 onnx_to_export.py --nemo_file /mnt/sangraha/venkat_shadeslayer/asr/momodels/stt_hi_conformer_ctc_medium.nemo --onnx_file /mnt/sangraha/venkat_shadeslayer/asr/momodels/mymodel3.onnx


finally:
python3 onnx_to_export.py 





Errors:

Running setup.py clean for hydra
Failed to build hydra
ERROR: Could not build wheels for hydra, which is required to install pyproject.toml-based projects

FIX:pip install hydra-core --upgrade

_____________________________________________________________________________________________________________________________________________________________________________________________________________________________________

STEP2: Transcribe text using both Nemo and Onnx models as required.

Get the manifest files from "https://objectstore.e2enetworks.net/indic-asr-public/indicwhisper/vistaar_benchmarks/indictts.zip"  
The above link contains wav files and their manifests in various languages like Hindi, Odia, Kannada and Gujarati to name a few.
You can use the above audio clips or provide your own audio clips to be transcribed.


pip install onnxruntime
pip install pyctcdecode
tar -xf /mnt/sangraha/venkat_shadeslayer/asr/momodels/stt_hi_conformer_ctc_medium.nemo 



