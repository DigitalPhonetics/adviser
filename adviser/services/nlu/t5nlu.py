import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig

###
### Taken from https://github.com/ConvLab/ConvLab-3/blob/04197837c424eb314bf47839a1a26f8a0048ff55/convlab/base_models/t5/nlu/serialization.py#L11
### License: Apache License 2.0
###
def deserialize_dialogue_acts(das_seq):
    dialogue_acts = []
    if len(das_seq) == 0:
        return dialogue_acts
    da_seqs = das_seq.split(']);[')  # will consume "])" and "["
    for i, da_seq in enumerate(da_seqs):
        if len(da_seq) == 0 or len(da_seq.split(']([')) != 2:
            continue
        if i == 0:
            if da_seq[0] == '[':
                da_seq = da_seq[1:]
        if i == len(da_seqs) - 1:
            if da_seq[-2:] == '])':
                da_seq = da_seq[:-2]
        
        try:
            intent_domain, slot_values = da_seq.split(']([')
            intent, domain = intent_domain.split('][')
        except:
            continue
        for slot_value in slot_values.split('],['):
            try:
                slot, value = slot_value.split('][')
            except:
                continue
            dialogue_acts.append({'intent': intent, 'domain': domain, 'slot': slot, 'value': value})
        
    return dialogue_acts


###
### Taken from https://github.com/ConvLab/ConvLab-3/blob/master/convlab/base_models/t5/nlu/nlu.py
###  License: Apache License 2.0
###
class T5NLU:
    def __init__(self, speaker, context_window_size, model_name_or_path, device='cuda'):
        assert speaker in ['user', 'system']
        self.speaker = speaker
        self.opponent = 'system' if speaker == 'user' else 'user'
        self.context_window_size = context_window_size
        self.use_context = context_window_size > 0
        
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=self.config)
        self.model.eval()
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
    def predict(self, utterance, context=list()):
        if self.use_context:
            if len(context) > 0 and type(context[0]) is list and len(context[0]) > 1:
                context = [item[1] for item in context]
            context = context[-self.context_window_size:]
            utts = context + [utterance]
        else:
            utts = [utterance]
        input_seq = '\n'.join([f"{self.opponent if (i % 2) == (len(utts) % 2) else self.speaker}: {utt}" for i, utt in enumerate(utts)])
        # print(input_seq)
        input_seq = self.tokenizer(input_seq, return_tensors="pt").to(self.device)
        # print(input_seq)
        output_seq = self.model.generate(**input_seq, max_length=256)
        # print(output_seq)
        output_seq = self.tokenizer.decode(output_seq[0], skip_special_tokens=True)
        # print(output_seq)
        das = deserialize_dialogue_acts(output_seq.strip())
        dialog_act = []
        for da in das:
            dialog_act.append([da['intent'], da['domain'], da['slot'], da.get('value','')])
        return dialog_act