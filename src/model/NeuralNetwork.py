import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math
from torchaudio.models.decoder import ctc_decoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class CRNN(nn.Module):
    def __init__(self, vocab=[], device='cuda'):
        super().__init__()
        self.vocab = vocab
        self.device = device

        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)


        for m in resnet.layer3:
            if m.downsample is not None:
                for layer in m.downsample:
                    if isinstance(layer, nn.Conv2d):
                        layer.stride = (2, 1)
            if m.conv1.stride == (2, 2):
                m.conv1.stride = (2, 1)
        for m in resnet.layer4:
            if m.downsample is not None:
                for layer in m.downsample:
                    if isinstance(layer, nn.Conv2d):
                        layer.stride = (2, 1)
            if m.conv1.stride == (2, 2):
                m.conv1.stride = (2, 1)

        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
        self.recurLayer = nn.LSTM(512, 256, num_layers=2, bidirectional=True, batch_first=True, device=device)
        self.fc = nn.Linear(512, len(vocab), device=device)
        
        self.decoder = ctc_decoder(
            lexicon= None,#"/mnt/d/ocr_lightnovel/model/LM/lexicon.txt",
            tokens=vocab,
            #lm="/mnt/d/ocr_lightnovel/model/LM/lightnovel.bin",
            blank_token="<blank>",
            sil_token=" ",
            beam_size=3,
            #lm_weight=0.5,
            #word_score=1
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.adaptive_pool(x)
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        x, _ = self.recurLayer(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=2)

    def ctc_decode_batch(self, log_probs):
        emissions = log_probs.detach().cpu().float().contiguous()
        beam_results = self.decoder(emissions)
        texts = []
        confs = []
        for b in range(len(beam_results)):
            if not beam_results[b]:
                texts.append(""); confs.append(0.0); continue
            best_hyp = beam_results[b][0]
            decoded_text = "".join([self.vocab[i] for i in best_hyp.tokens])
            decoded_text = decoded_text.replace("<blank>", "").strip()
            num_tokens = len(best_hyp.tokens)
            score = best_hyp.score
            conf = math.exp(score / num_tokens) if num_tokens > 0 else 0.0  
            texts.append(decoded_text)
            confs.append(min(conf, 1.0))      
        return texts, confs

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True

    def freeze_cnn(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        