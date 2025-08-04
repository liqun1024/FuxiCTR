import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import T5ForConditionalGeneration, T5Config
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput


class GenRecMultiHead(T5ForConditionalGeneration):
    """
    GenRec model for ranking tasks, extending T5ForConditionalGeneration.
    Supports multiple LM heads for different vocab sizes.
    Allows setting a temperature for the ranking loss.
    """

    def __init__(self, 
                 token_level_vocab_sizes: list[int], 
                 special_vocab_size: int = 10,
                 loss_temperature: float = 1.0,
                 config: T5Config = None):
        super().__init__(config)
        self.token_level_vocab_sizes = token_level_vocab_sizes
        self.special_vocab_size = special_vocab_size

        self.embeddings = nn.Embedding(
            special_vocab_size + sum(token_level_vocab_sizes), 
            config.d_model
        )

        self.loss_temperature = loss_temperature
        
        self.lm_heads = nn.ModuleList(
            [nn.Linear(config.d_model, vocab_size, bias=False) for vocab_size in token_level_vocab_sizes]
        )
        self.lm_head = None


        self.level_offsets = [0] * len(token_level_vocab_sizes)
        cumulative_offset = special_vocab_size
        for i, size in enumerate(token_level_vocab_sizes):
            self.level_offsets[i] = cumulative_offset
            cumulative_offset += size

        if self.config.tie_word_embeddings:
            print("WARNING: tie_word_embeddings is not supported with Multi LM heads. Disabling.")
            self.config.tie_word_embeddings = False

    def ranking_loss(self, lm_logits, labels):
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            losses = []
            token_levels = len(self.token_level_vocab_sizes)

            for i in range(token_levels):
                head_labels = labels[:, i::token_levels].clone()
                head_labels[head_labels == 0] = -100
                head_labels[head_labels != -100] -= self.level_offsets[i] # Adjust labels to match the token level offset
                logits = lm_logits[i::token_levels]
                if logits is not None and head_labels.numel() > 0:
                    head_logits = torch.stack(logits, dim=1)  # [batch, seq, vocab_size]
                    t_logits = head_logits / self.loss_temperature
                    loss = loss_fct(
                        t_logits.reshape(-1, t_logits.size(-1)),
                        head_labels.reshape(-1)
                    )
                    losses.append(loss)
            if losses:
                return sum(losses) / len(losses)
        return None

    def total_loss(self, lm_logits, labels):
        loss = self.ranking_loss(lm_logits, labels)
        return loss

    def forward(
        self,
        input_ids=None, 
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        inputs_embeds = self.embeddings(input_ids)

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(labels)
        decoder_inputs_embeds = self.embeddings(decoder_input_ids)

        decoder_outputs = self.decoder(
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0] # (batch_size, sequence_length, model_dim)

        sequence_len = sequence_output.size(1)
        token_levels = len(self.token_level_vocab_sizes)
        all_logits = [None] * sequence_len

        for i in range(token_levels):
            sub_sequence_hidden_states = sequence_output[:, i::token_levels, :]

            if sub_sequence_hidden_states.size(1) > 0:
                sub_sequence_logits = self.lm_heads[i](sub_sequence_hidden_states)
                for j, original_idx in enumerate(range(i, sequence_len, token_levels)):
                    all_logits[original_idx] = sub_sequence_logits[:, j, :]

        lm_logits = all_logits # (sequence_length, batch_size, vocab_size)

        loss = None
        if labels is not None:
            loss = self.total_loss(lm_logits, labels)

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=None,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 20
    ):
        """
        Generates token sequences autoregressively for the GenRec model.
        This is a simplified implementation using greedy search.

        Args:
            input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                The input sequence IDs.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Mask to avoid performing attention on padding token indices.
            max_length (`int`, *optional*, defaults to 20):
                The maximum length of the sequence to be generated.
        
        Returns:
            `torch.Tensor` of shape `(batch_size, sequence_length)`:
                The generated sequence of token IDs.
        """
        self.eval()

        batch_size = input_ids.shape[0]
        device = input_ids.device

        inputs_embeds = self.embeddings(input_ids)  # (batch_size, sequence_length, d_model)

        encoder_outputs = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state

        decoder_start_ids = torch.full(
            (batch_size, 1), self.config.decoder_start_token_id, dtype=torch.long, device=device
        )
        decoder_inputs_embeds = self.embeddings(decoder_start_ids)  # (batch_size, 1, d_model)

        generated_ids = torch.empty((batch_size, 0), dtype=torch.long, device=device)

        past_key_values = None

        token_levels = len(self.token_level_vocab_sizes)
        for step in range(max_length):
            token_level_idx = step % token_levels

            decoder_outputs = self.decoder(
                inputs_embeds=decoder_inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            
            last_hidden_state = decoder_outputs.last_hidden_state # (batch_size, 1, d_model)
            
            lm_head = self.lm_heads[token_level_idx]
            logits = lm_head(last_hidden_state) # (batch_size, 1, vocab_size)
            
            next_token_logits = logits.squeeze(1) # (batch_size, vocab_size)
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1) # (batch_size, 1)
            next_token_id = self.level_offsets[token_level_idx] + next_token_id

            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

            decoder_inputs_embeds = self.embeddings(next_token_id) # (batch_size, 1, d_model)
            past_key_values = decoder_outputs.past_key_values

        return generated_ids


if __name__ == '__main__':
    token_level_vocab_sizes = [10, 10, 10, 100]
    token_levels = len(token_level_vocab_sizes)
    special_vocab_size = 10
    model_dim = 128

    config = T5Config(
        vocab_size=1, 
        d_model=model_dim,
        d_kv=model_dim // 2,
        d_ff=model_dim * 2,
        num_layers=2,
        token_levels=4,
        decoder_start_token_id=1
    )

    model = GenRecMultiHead(
        config=config,
        token_level_vocab_sizes=token_level_vocab_sizes,
        special_vocab_size=special_vocab_size
    )

    # token i of item: special_vocab_size + sum(token_level_vocab_sizes[:i]) + idx
    input_ids = torch.tensor([[13, 25, 37, 17, 25, 34, 0, 11, 22, 33], [13, 26, 32, 15, 24, 32, 0, 11, 24, 32]], dtype=torch.long)
    labels = torch.tensor([[11, 22, 33, 78, 13, 25, 37, 90], [12, 24, 32, 69, 13, 22, 32, 52]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Labels shape: {labels.shape}\n")


    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    print(f"Loss: {outputs.loss.item()}")
    

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=8
        )
    print(f"Generated IDs: {generated_ids}")