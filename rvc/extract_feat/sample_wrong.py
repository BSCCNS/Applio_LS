
import torch
import numpy as np
from scipy import signal
            
def pipeline(self, model, audio):

    audio = signal.filtfilt(bh, ah, audio)

    audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
    opt_ts = []
    if audio_pad.shape[0] > self.t_max:
        audio_sum = np.zeros_like(audio)
        for i in range(self.window):
            audio_sum += audio_pad[i : i - self.window]
        for t in range(self.t_center, audio.shape[0], self.t_center):
            opt_ts.append(
                t
                - self.t_query
                + np.where(
                    np.abs(audio_sum[t - self.t_query : t + self.t_query])
                    == np.abs(audio_sum[t - self.t_query : t + self.t_query]).min()
                )[0][0]
            )
    s = 0
    t = None
    audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
    
    for t in opt_ts:
        t = t // self.window * self.window
        self.extract_feats(
                model,
                audio_pad[s : t + self.t_pad2 + self.window])
        s = t
    self.extract_feats(
        model,
        audio_pad[t:]) 
    
    return None

def extract_feats(self, model, audio):
    with torch.no_grad():
        audio_torch = torch.from_numpy(audio).float()
        audio_torch = audio_torch.mean(-1) if audio_torch.dim() == 2 else audio_torch
        assert audio_torch.dim() == 1, audio_torch.dim()
        audio_torch = audio_torch.view(1, -1).to(self.device)

        feats = model(audio_torch)["last_hidden_state"]
        return  feats