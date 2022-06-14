# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict, List, Optional, Tuple
from fairseq.models.xlmt_decoder_variant import XLMTAddFFN
import torch
import torch.nn as nn
from torch import Tensor
from fairseq.models.transformer_from_pretrained_infoxlm import TransformerEncoderFromPretrainedInfoXLM, \
    TransformerFromPretrainedInfoXLMModel
from fairseq.models.xlmt_decoder_variant import upgrade_state_dict_for_two_ffn
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    TransformerDecoder,
    TransformerEncoder,
    TransformerModel,
    base_architecture as transformer_base_architecture,
)
from fairseq.modules.multihead_attention import MultiheadAttention
from fairseq.modules import LayerNorm, TransformerDecoderLayer
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from fairseq import utils
from fairseq.file_io import PathManager
import logging

logger = logging.getLogger(__name__)


def expand_embedding_matrix(state, encoder):
    if state['model']['encoder.embed_tokens.weight'].size(0) < encoder.embed_tokens.weight.size(0):
        offset = encoder.embed_tokens.weight.size(0) - state['model']['encoder.embed_tokens.weight'].size(0)
        embed_dim = encoder.embed_tokens.weight.size(1)
        state['model']['encoder.embed_tokens.weight'] = torch.cat([state['model']['encoder.embed_tokens.weight'],
                                                                   state['model']['encoder.embed_tokens.weight'].new(
                                                                       offset, embed_dim).fill_(0)], dim=0)
        state['model']['decoder.embed_tokens.weight'] = state['model']['encoder.embed_tokens.weight']
        state['model']['decoder.output_projection.weight'] = state['model']['encoder.embed_tokens.weight']
        logger.info("Expanding the embedding matrix to match the current shape...")
    return state


def upgrade_mt_state_for_adapter_model(
        state_dict: Dict[str, Any], pretrained_mt_checkpoint: str, encoder: TransformerEncoder
) -> Dict[str, Any]:
    if not os.path.exists(pretrained_mt_checkpoint):
        raise IOError("Model file not found: {}".format(pretrained_mt_checkpoint))

    # state = checkpoint_utils.load_checkpoint_to_cpu(pretrained_infoxlm_checkpoint)
    with open(PathManager.get_local_path(pretrained_mt_checkpoint), "rb") as f:
        state = torch.load(f, map_location=torch.device("cpu"))
    state = expand_embedding_matrix(state, encoder)
    mt_state_dict = state["model"]
    for key in mt_state_dict.keys():
        state_dict[key] = mt_state_dict[key]
    return state_dict


@register_model("language_specific_transformer")
class LanguageSpecificTransformerModel(TransformerFromPretrainedInfoXLMModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerFromPretrainedInfoXLMModel.add_args(parser)
        parser.add_argument(
            "--variant",
            type=str,
            metavar="STR",
        )
        parser.add_argument(
            "--pretrained-mt-checkpoint",
            type=str,
            metavar="STR",
        )
        parser.add_argument(
            "--use-adapter",
            action="store_true",
        )
        parser.add_argument(
            "--high-langs",
            type=str,
            metavar="STR",
        )
        parser.add_argument(
            "--adapter-dim",
            type=int,
        )
        parser.add_argument(
            "--adapter-dropout",
            type=float,
        )
        parser.add_argument(
            "--freeze-embedding",
            action="store_true",
        )
        parser.add_argument(
            "--freeze-encoder",
            action="store_true",
        )
        parser.add_argument(
            "--freeze-decoder",
            action="store_true",
        )
        parser.add_argument(
            "--freeze-decoder-without-cross-attention",
            action="store_true",
        )

    @classmethod
    def build_encoder(cls, args, tgt_dict, embed_tokens):
        return LanguageSpecificTransformerEncoder(args, tgt_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return LanguageSpecificTransformerDecoder(args, tgt_dict, embed_tokens)

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

    def forward(
            self,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            return_all_hiddens: bool = True,
            features_only: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            **extra_args
    ):
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens,
            src_lang_id=extra_args["src_lang_id"] if "src_lang_id" in extra_args.keys() else None,
            tgt_lang_id=extra_args["tgt_lang_id"] if "tgt_lang_id" in extra_args.keys() else None,
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            src_lang_id=extra_args["src_lang_id"] if "src_lang_id" in extra_args.keys() else None,
            tgt_lang_id=extra_args["tgt_lang_id"] if "tgt_lang_id" in extra_args.keys() else None,
        )
        return decoder_out


class AdapterLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, 'quant_noise_pq', 0)
        self.quant_noise_block_size = getattr(args, 'quant_noise_pq_block_size', 8) or 8
        self.adapter_dropout = getattr(args, "adapter_dropout", args.dropout)
        self.dropout_module = FairseqDropout(
            self.adapter_dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu') or "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.decoder_normalize_before
        self.adapter_dim = getattr(args, "adapter_dim", self.embed_dim // 4)
        self.fc1 = self.build_fc1(
            self.embed_dim,
            self.adapter_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            self.adapter_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def forward(self, x):
        residual = x
        x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = x + residual
        return x


class LanguageSpecificTransformerEncoder(TransformerEncoderFromPretrainedInfoXLM):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        if getattr(args, 'freeze_encoder', False):
            for param in self.layers.parameters():
                param.requires_grad = False

    def get_lang_id(self, lang_id):
        # assert lang_id is not None, "lang_id in Batch can not be None Type !"
        if isinstance(lang_id, int):
            return lang_id
        else:
            return int(lang_id[0].cpu())

    def forward(
            self,
            src_tokens,
            src_lengths,
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
            src_lang_id=None,
            tgt_lang_id=None,
            **kwargs
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings
        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        encoder_states = []

        # encoder layers
        for idx, layer in enumerate(self.layers):
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }


class LanguageSpecificTransformerDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        args.pretrained_infoxlm_checkpoint = getattr(args, "pretrained_infoxlm_checkpoint", "")
        if os.path.exists(args.pretrained_infoxlm_checkpoint):
            if args.variant == 'addffn':
                infoxlm_loaded_state_dict = upgrade_state_dict_for_two_ffn(
                    state_dict=self.state_dict(),
                    pretrained_infoxlm_checkpoint=args.pretrained_infoxlm_checkpoint,
                    num_layers=args.decoder_layers,
                )
                logger.info("Loading decoder from {0}".format(args.pretrained_infoxlm_checkpoint))
            self.load_state_dict(infoxlm_loaded_state_dict, strict=False)
        self.use_adapter = args.use_adapter
        self.high_langs = args.high_langs.split(',')
        self.langs = args.langs
        if self.use_adapter:
            self.adapter = nn.ModuleList([])
            self.adapter.extend(
                nn.ModuleList(
                    [
                        AdapterLayer(args)
                        for _ in range(len(self.high_langs))
                    ])
            )
        else:
            self.adapter = None
        if getattr(args, 'freeze_decoder', False):
            for param in self.layers.parameters():
                param.requires_grad = False
        if getattr(args, 'freeze_decoder_without_cross_attention', False):
            for i in range(args.decoder_layers):
                for param in self.layers[i].parameters():
                    param.requires_grad = False
                for param in self.layers[i].encoder_attn.parameters():
                    param.requires_grad = True
                for param in self.layers[i].encoder_attn_layer_norm.parameters():
                    param.requires_grad = True
        if getattr(args, 'freeze_adapter', False) and getattr(args, "use_adapter", False):
            for param in self.adapter.parameters():
                param.requires_grad = False
        if getattr(args, 'freeze_embedding', False):
            for param in self.embed_tokens.parameters():
                param.requires_grad = False

    def build_decoder_layer(self, args, no_encoder_attn=False):
        if args.variant == 'addffn':
            layer = XLMTAddFFN(args, no_encoder_attn)
        else:
            layer = TransformerDecoderLayer(args, no_encoder_attn)
        if getattr(args, "checkpoint_activations", False):
            layer = checkpoint_wrapper(layer)
        return layer

    def get_lang_id(self, lang_id):
        # assert lang_id is not None, "lang_id in Batch can not be None Type !"
        if isinstance(lang_id, int):
            return lang_id
        else:
            return int(lang_id[0].cpu())

    def forward(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            features_only: bool = False,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            src_lengths: Optional[Any] = None,
            return_all_hiddens: bool = False,
            src_lang_id=None,
            tgt_lang_id=None
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lang_id=src_lang_id,
            tgt_lang_id=tgt_lang_id,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]],
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            src_lang_id=None,
            tgt_lang_id=None
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
            src_lang_id=src_lang_id,
            tgt_lang_id=tgt_lang_id
        )

    def extract_features_scriptable(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]],
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            src_lang_id=None,
            tgt_lang_id=None
    ):
        """
        Similar to *forward* but only return features.
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).
        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                        encoder_out is not None
                        and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if self.use_adapter:
            tgt_lang_id = self.get_lang_id(tgt_lang_id) - 1
            if self.langs[tgt_lang_id] in self.high_langs:
                adapter_id = self.high_langs.index(self.langs[tgt_lang_id])
                x = self.adapter[adapter_id](x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}


@register_model_architecture(
    "language_specific_transformer", "language_specific_xlmt"
)
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)
    args.layer_wise_attention = getattr(args, "layer_wise_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)

    args.init_encoder_only = getattr(args, "init_encoder_only", False)
    args.init_decoder_only = getattr(args, "init_decoder_only", False)
    args.max_positions = getattr(args, "max_positions", 512)
    #
    args.adapter_dim = getattr(args, "adapter_dim", args.decoder_embed_dim // 4)
    args.use_adapter = getattr(args, "use_adapter", False)


@register_model_architecture(
    "language_specific_transformer", "language_specific_transformer"
)
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)

    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)
    #
    args.adapter_dim = getattr(args, "adapter_dim", args.decoder_embed_dim // 4)
    args.use_adapter = getattr(args, "use_adapter", False)


@register_model_architecture(
    "language_specific_transformer", "language_specific_transformer_big"
)
def adapter_transformer_large(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    base_architecture(args)
