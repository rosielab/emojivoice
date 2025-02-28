import datetime as dt
import math
import random

import torch

import matcha.utils.monotonic_align as monotonic_align
from matcha import utils
from matcha.models.baselightningmodule import BaseLightningClass
from matcha.models.components.flow_matching import CFM
from matcha.models.components.text_encoder import TextEncoder
from matcha.utils.model import (
    denormalize,
    duration_loss,
    fix_len_compatibility,
    generate_path,
    sequence_mask,
)

WORD_STRETCH = 1.8
WORD_COMPRESS = 1

log = utils.get_pylogger(__name__)


class MatchaTTS(BaseLightningClass):  # 🍵
    def __init__(
        self,
        n_vocab,
        n_spks,
        spk_emb_dim,
        n_feats,
        encoder,
        decoder,
        cfm,
        data_statistics,
        out_size,
        optimizer=None,
        scheduler=None,
        prior_loss=True,
        use_precomputed_durations=False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.n_vocab = n_vocab
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.n_feats = n_feats
        self.out_size = out_size
        self.prior_loss = prior_loss
        self.use_precomputed_durations = use_precomputed_durations

        if n_spks > 1:
            self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim)

        self.encoder = TextEncoder(
            encoder.encoder_type,
            encoder.encoder_params,
            encoder.duration_predictor_params,
            n_vocab,
            n_spks,
            spk_emb_dim,
        )

        self.decoder = CFM(
            in_channels=2 * encoder.encoder_params.n_feats,
            out_channel=encoder.encoder_params.n_feats,
            cfm_params=cfm,
            decoder_params=decoder,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )

        self.update_data_statistics(data_statistics)
    
    def clarity_adjust(self, x, device):
        target_vowels = [51, 102, 135, 63, 138, 69]
        word_dict = {}
        start = 0
        phonemes = x[0]
        clarity_dict = {}

        updated_phonemes = []
        offset = 0

        for index, value in enumerate(phonemes):
            if value == 16:
                word_dict[start] = phonemes[start:index]
                start = index + 1

        # Add the remaining part of the list (after the last 16)
        if start < len(phonemes):
            word_dict[start] = phonemes[start:]

        for word_index in word_dict:
            word = word_dict[word_index].tolist()
            tense = False
            lax = False
            both = False
            punctuation = False
            # detect the clarity flag
            if word[1] == 5:
                # if not before punctuation
                if word[-2] == 5:
                    word = word[2:-2]
                # followed by punctuation
                elif word[-4] ==  5:
                    print('heyo')
                    del word[2]
                    del word[-3]
                    del word[-3]
                    punctuation = True
                for i, letter in enumerate(word):
                    if letter in target_vowels and word[i-2] == 156:
                        #has the primary stress
                        if letter == 51 or letter == 63 or letter == 69:
                            tense = True
                            lax = False
                            break
                        else:
                            lax = True
                            tense = False
                            break
                    if both != True:
                        # is ɪ and not joined into a dipthong, and is not in "ing"
                        if (
                            letter == 102 and
                            word[i-2] != 43 and
                            word[i-2] != 47 and
                            word[i-2] != 76
                        ):
                            if (i+2 < len(word)):
                                if word[i+2] != 112:
                                    if tense == False:
                                        lax = True
                                    #tense and lax vowel in one word
                                    else: # issue here because will stop searching and maybe primary stress is later
                                        tense = False
                                        lax = False
                                        both = True
                        # is ʊ and is not joined into a dipthong
                        elif letter == 135 and word[i-2] != 43 and word[i-2] != 57:
                            if tense == False:
                                lax == True
                            else: # again need to fix this
                                tense = False
                                lax = False
                                both = True
                        # is ʌ
                        elif letter == 138:
                            if tense == False:
                                lax = True
                            else:
                                tense = False
                                lax = False
                                both = True
                        # is i and not at the end of a word, or is u or is ɑ
                        elif letter == 69 or letter == 63 or (letter == 51 and i != len(word)-1):
                            if lax == False:
                                tense == True
                            else:
                                tense = False
                                lax = False
                                both = True
                adjusted_index = word_index - offset
                if tense:
                    clarity_dict[word_index - offset] = ["tense", len(word)]
                if lax:
                    clarity_dict[word_index - offset] = ["lax", len(word)]
                offset += 2
                if punctuation:
                    offset += 1
            if updated_phonemes:
                updated_phonemes.append(16)  # Insert separator between words
            updated_phonemes.extend(word)

        if updated_phonemes:
            clarity_array = [1] * len(updated_phonemes)
        else:
            clarity_array = [1] * len(phonemes)

        if clarity_dict:
            last_index = 0

            for idx, index in enumerate(clarity_dict):
                type_, count = clarity_dict[index]
                count = int(count)
                next_index = (
                    list(clarity_dict.keys())[idx + 1] if idx + 1 < len(clarity_dict) else len(clarity_array)
                )

                # Calculate available space for ramps
                space_after = next_index - (index + count)
                ramp_length_before = min(6, index - last_index)
                ramp_length_after = min(6, space_after)

                if type_ == "tense":
                    # Gradually increase from 1 to WORD_STRETCH
                    for i in range(index - ramp_length_before, index):
                        clarity_array[i] = 1 + (WORD_STRETCH - 1) * (i - (index - ramp_length_before)) / ramp_length_before

                    # Assign WORD_STRETCH to the core region
                    clarity_array[index:index + count] = [WORD_STRETCH] * count

                    # Gradually decrease from WORD_STRETCH to 1
                    for j in range(index + count, index + count + ramp_length_after):
                        clarity_array[j] = WORD_STRETCH - (WORD_STRETCH - 1) * (j - (index + count)) / ramp_length_after

                elif type_ == "lax":
                    # Gradually increase from 1 to WORD_COMPRESS
                    for i in range(index - ramp_length_before, index):
                        clarity_array[i] = 1 + (WORD_COMPRESS - 1) * (i - (index - ramp_length_before)) / ramp_length_before

                    # Assign WORD_COMPRESS to the core region
                    clarity_array[index:index + count] = [WORD_COMPRESS] * count
    
                    # Gradually decrease from WORD_COMPRESS to 1
                    for j in range(index + count, index + count + ramp_length_after):
                        clarity_array[j] = WORD_COMPRESS - (WORD_COMPRESS - 1) * (j - (index + count)) / ramp_length_after
    
                # Update last_index
                last_index = index + count + ramp_length_after

            # Assign trailing 1s for the rest of the array
            clarity_array[last_index:] = [1] * (len(updated_phonemes) - last_index)
 
        return torch.tensor(clarity_array,dtype=torch.float, device=device), torch.tensor([updated_phonemes],dtype=torch.long, device=device)

    @torch.inference_mode()
    def synthesise(self, x, x_lengths, n_timesteps, temperature=1.0, spks=None, length_scale=1.0, clarity=False):
        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            spks (bool, optional): speaker ids.
                shape: (batch_size,)
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.

        Returns:
            dict: {
                "encoder_outputs": torch.Tensor, shape: (batch_size, n_feats, max_mel_length),
                # Average mel spectrogram generated by the encoder
                "decoder_outputs": torch.Tensor, shape: (batch_size, n_feats, max_mel_length),
                # Refined mel spectrogram improved by the CFM
                "attn": torch.Tensor, shape: (batch_size, max_text_length, max_mel_length),
                # Alignment map between text and mel spectrogram
                "mel": torch.Tensor, shape: (batch_size, n_feats, max_mel_length),
                # Denormalized mel spectrogram
                "mel_lengths": torch.Tensor, shape: (batch_size,),
                # Lengths of mel spectrograms
                "rtf": float,
                # Real-time factor
        """
        clarity_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # For RTF computation
        t = dt.datetime.now()

        if self.n_spks > 1:
            # Get speaker embedding
            spks = self.spk_emb(spks.long())

        if clarity:
            clarity_array, x = self.clarity_adjust(x, clarity_device)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spks)
        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        if clarity:
            w_ceil = w_ceil * clarity_array
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = y_lengths.max()
        y_max_length_ = fix_len_compatibility(y_max_length)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # Align encoded text and get mu_y
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        encoder_outputs = mu_y[:, :, :y_max_length]

        # Generate sample tracing the probability flow
        decoder_outputs = self.decoder(mu_y, y_mask, n_timesteps, temperature, spks)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        t = (dt.datetime.now() - t).total_seconds()
        rtf = t * 22050 / (decoder_outputs.shape[-1] * 256)

        return {
            "encoder_outputs": encoder_outputs,
            "decoder_outputs": decoder_outputs,
            "attn": attn[:, :, :y_max_length],
            "mel": denormalize(decoder_outputs, self.mel_mean, self.mel_std),
            "mel_lengths": y_lengths,
            "rtf": rtf,
        }       

    def forward(self, x, x_lengths, y, y_lengths, spks=None, out_size=None, cond=None, durations=None):
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotinic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. flow matching loss: loss between mel-spectrogram and decoder outputs.

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            y (torch.Tensor): batch of corresponding mel-spectrograms.
                shape: (batch_size, n_feats, max_mel_length)
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
                shape: (batch_size,)
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
            spks (torch.Tensor, optional): speaker ids.
                shape: (batch_size,)
        """
        if self.n_spks > 1:
            # Get speaker embedding
            spks = self.spk_emb(spks)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spks)
        y_max_length = y.shape[-1]

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        if self.use_precomputed_durations:
            attn = generate_path(durations.squeeze(1), attn_mask.squeeze(1))
        else:
            # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
            with torch.no_grad():
                const = -0.5 * math.log(2 * math.pi) * self.n_feats
                factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
                y_square = torch.matmul(factor.transpose(1, 2), y**2)
                y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
                mu_square = torch.sum(factor * (mu_x**2), 1).unsqueeze(-1)
                log_prior = y_square - y_mu_double + mu_square + const

                attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
                attn = attn.detach()  # b, t_text, T_mel

        # Compute loss between predicted log-scaled durations and those obtained from MAS
        # refered to as prior loss in the paper
        logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        dur_loss = duration_loss(logw, logw_, x_lengths)

        # Cut a small segment of mel-spectrogram in order to increase batch size
        #   - "Hack" taken from Grad-TTS, in case of Grad-TTS, we cannot train batch size 32 on a 24GB GPU without it
        #   - Do not need this hack for Matcha-TTS, but it works with it as well
        if not isinstance(out_size, type(None)):
            max_offset = (y_lengths - out_size).clamp(0)
            offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
            out_offset = torch.LongTensor(
                [torch.tensor(random.choice(range(start, end)) if end > start else 0) for start, end in offset_ranges]
            ).to(y_lengths)
            attn_cut = torch.zeros(attn.shape[0], attn.shape[1], out_size, dtype=attn.dtype, device=attn.device)
            y_cut = torch.zeros(y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device)

            y_cut_lengths = []
            for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
                y_cut_length = out_size + (y_lengths[i] - out_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]

            y_cut_lengths = torch.LongTensor(y_cut_lengths)
            y_cut_mask = sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)

            attn = attn_cut
            y = y_cut
            y_mask = y_cut_mask

        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)

        # Compute loss of the decoder
        diff_loss, _ = self.decoder.compute_loss(x1=y, mask=y_mask, mu=mu_y, spks=spks, cond=cond)

        if self.prior_loss:
            prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
            prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
        else:
            prior_loss = 0

        return dur_loss, prior_loss, diff_loss, attn