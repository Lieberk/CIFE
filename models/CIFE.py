from transformers.models.bart.modeling_bart import *
import torch.nn as nn
import torch
import math
from transformers.modeling_outputs import BaseModelOutput


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.eye(768))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1

        output = torch.matmul(adj.float(), hidden.float()) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class Guided_Attention(nn.Module):

    def __init__(self, hidden_size, num_attention_heads, dropout_prob):

        super(Guided_Attention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = int(self.num_attention_heads * self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, G_hidden_states):
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (1.0 - attention_mask) * -10000.0

        mixed_query_layer = self.query(G_hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class CIFE(nn.Module):
    def __init__(self, cfg, tkr):
        super(CIFE, self).__init__()
        self.cfg = cfg
        self.tkr = tkr
        self.text_dim = 768
        self.model_gen = BartForConditionalGeneration.from_pretrained(".\dataset\Pretrain/bart-base")
        self.Gatt = Guided_Attention(hidden_size=self.text_dim, num_attention_heads=8, dropout_prob=0.3)
        self.gc = GraphConvolution(768, 768)

    def forward(self, mode='train', **kwargs):
        target_ids = kwargs['target_ids']
        graph = kwargs['model_adj']

        fea_title = kwargs['title']
        title_mask = kwargs['title_mask']

        audio_transcript = kwargs['audio_transcript']
        audio_transcript_mask = kwargs['audio_transcript_mask']

        ivod_feature = kwargs['ivod_feature']
        ivod_mask = kwargs['ivod_mask']

        concat_feat = torch.cat([fea_title, ivod_feature, audio_transcript], dim=1)
        concat_mask = torch.cat([title_mask, ivod_mask, audio_transcript_mask], dim=1)

        context_enc_out = self.model_gen.get_encoder()(inputs_embeds=concat_feat, attention_mask=concat_mask)
        context_enc_feat = context_enc_out.last_hidden_state
        x = self.gc(context_enc_feat, graph)

        ieam_feature = kwargs['ieam_feature']
        ieam_mask = kwargs['ieam_mask']

        gen_mask = concat_mask
        gen_feat = context_enc_feat + x
        gen_feat = gen_feat + self.Gatt(ieam_feature, ieam_mask, gen_feat)
        enc_output = BaseModelOutput(last_hidden_state=gen_feat)

        if mode == 'train':
            gen = self.model_gen(encoder_outputs=enc_output, attention_mask=gen_mask, labels=target_ids)
            return gen

        elif mode == 'eval' or mode == 'gen':
            with torch.no_grad():
                if mode == 'eval':
                    gen = self.model_gen(encoder_outputs=enc_output, attention_mask=gen_mask, labels=target_ids)
                    return gen.loss

                elif mode == 'gen':
                    generation_cfgs = {"max_length": self.cfg.eval.eval_max_len,
                                       "min_length": self.cfg.eval.eval_min_len,
                                       "pad_token_id": self.tkr.pad_token_id,
                                       'eos_token_id': self.tkr.eos_token_id, "num_beams": self.cfg.eval.num_beams,
                                       'top_p': self.cfg.eval.top_p, 'top_k': self.cfg.eval.top_k,
                                       'temperature': self.cfg.eval.temperature, 'do_sample': True,
                                       'repetition_penalty': self.cfg.eval.repetition_penalty,
                                       'no_repeat_ngram_size': self.cfg.eval.no_repeat_ngram_size}

                    gen_result = self.model_gen.generate(encoder_outputs=enc_output,
                                                         attention_mask=gen_mask,
                                                         **generation_cfgs)

                    gen_decoded = self.tkr.batch_decode(gen_result, skip_special_tokens=True)

                    return gen_decoded
                return None

        else:
            raise ValueError('Mode should be among [train, eval, gen].')