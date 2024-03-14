import nemo.collections.asr as nemo_asr

# Load ASR model
asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_hi_conformer_ctc_medium")

# from nemo.core.classes import ModelPT, Exportable

# Deriving from Exportable
# class MyExportableModel(ModelPT, Exportable):
#     pass

# Create an instance of the model using the correct name (asr_model)
# mymodel = MyExportableModel.from_pretrained(model_name="stt_hi_conformer_ctc_medium")
asr_model.eval()
asr_model.to('cuda')  # or to('cpu') if you don't have GPU

# Export pre-trained model to ONNX file for deployment
asr_model.export('mymodel2.onnx')
