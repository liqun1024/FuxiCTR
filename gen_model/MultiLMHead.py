import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import T5ForConditionalGeneration, T5Config
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput


class GenRec(T5ForConditionalGeneration):
    """
    GenRec model for ranking tasks, extending T5ForConditionalGeneration.
    Supports multiple LM heads for different vocab sizes.
    Allows setting a temperature for the ranking loss.
    """

    def __init__(self, 
                 lm_heads_vocab_sizes, special_vocab_size=10,
                 loss_temperature=1.0,
                 config: T5Config = None):
        super().__init__(config)

        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, config.d_model) for vocab_size in lm_heads_vocab_sizes]
        )
        self.special_embedding = nn.Embedding(special_vocab_size, config.d_model)

        self.loss_temperature = loss_temperature

        if self.config.tie_word_embeddings:
            print("WARNING: tie_word_embeddings is not supported with Multi LM heads. Disabling.")
            self.config.tie_word_embeddings = False
            
        self.lm_heads = nn.ModuleList(
            [nn.Linear(config.d_model, vocab_size, bias=False) for vocab_size in lm_heads_vocab_sizes]
        )
        self.lm_head = None

    def ranking_loss(self, lm_logits, labels):
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            losses = []
            num_heads = len(self.lm_heads)

            for i in range(num_heads):
                head_labels = labels[:, i::num_heads]
                logits = lm_logits[i::num_heads]
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

    def get_embeddings(self, input_ids, use_last_embedding):
        """
        Get embeddings for the input_ids using the appropriate embedding layer.
        This method handles special tokens separately and groups only the regular tokens.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        inputs_embeds = torch.zeros(batch_size, seq_len, self.config.d_model, device=device)

        # Special tokens handling
        special_mask = input_ids < 0
        if special_mask.any():
            special_ids = input_ids[special_mask]
            special_embed_idx = (-special_ids - 1).long()
            special_embeds = self.special_embedding(special_embed_idx)
            inputs_embeds[special_mask] = special_embeds

        # Group regular tokens
        regular_mask = ~special_mask
        regular_batch_indices, regular_seq_indices = torch.where(regular_mask)
        regular_ids = input_ids[regular_mask]

        if regular_ids.numel() == 0:
            return inputs_embeds
        
        if use_last_embedding:
            num_groups = len(self.embeddings)  # Use all embeddings including the last one
        else:
            num_groups = len(self.embeddings) - 1  # Exclude the last embedding which is for the decoder

        group_assignments = torch.arange(regular_ids.numel(), device=device) % num_groups
        all_regular_embeds = torch.zeros(regular_ids.numel(), self.config.d_model, device=device)
        for i in range(num_groups):
            mask_i = (group_assignments == i)
            if not mask_i.any():
                continue
            ids_i = regular_ids[mask_i]
            embeds_i = self.embeddings[i](ids_i)
            all_regular_embeds[mask_i] = embeds_i

        inputs_embeds[regular_batch_indices, regular_seq_indices] = all_regular_embeds

        return inputs_embeds

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
        **kwargs,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        inputs_embeds = self.get_embeddings(input_ids, use_last_embedding=False)

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

        decoder_start_token = torch.full(
            (labels.size(0), 1), 
            self.config.decoder_start_token_id, 
            dtype=labels.dtype, device=labels.device
        )
        decoder_input_ids = torch.cat([decoder_start_token, labels], dim=1)
        decoder_inputs_embeds = self.get_embeddings(decoder_input_ids, use_last_embedding=True)
        decoder_inputs_embeds = decoder_inputs_embeds[:, :-1, :]

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
        num_heads = len(self.lm_heads)
        all_logits = [None] * sequence_len

        for i in range(num_heads):
            sub_sequence_hidden_states = sequence_output[:, i::num_heads, :]

            if sub_sequence_hidden_states.size(1) > 0:
                sub_sequence_logits = self.lm_heads[i](sub_sequence_hidden_states)
                for j, original_idx in enumerate(range(i, sequence_len, num_heads)):
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
        max_length: int = 20,
        **kwargs,
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

        inputs_embeds = self.get_embeddings(input_ids, use_last_embedding=False)

        encoder_outputs = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state

        decoder_start_token_id = self.config.decoder_start_token_id
        generated_ids = torch.full(
            (batch_size, 1), decoder_start_token_id, dtype=torch.long, device=device
        )

        decoder_inputs_embeds = self.get_embeddings(generated_ids, use_last_embedding=True)
        past_key_values = None
        num_heads = len(self.lm_heads)
        for step in range(max_length - 1):
            current_head_idx = step % num_heads

            decoder_outputs = self.decoder(
                inputs_embeds=decoder_inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            
            last_hidden_state = decoder_outputs.last_hidden_state # (batch_size, 1, d_model)
            
            lm_head = self.lm_heads[current_head_idx]
            logits = lm_head(last_hidden_state) # (batch_size, 1, vocab_size)
            
            next_token_logits = logits.squeeze(1) # (batch_size, vocab_size)
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1) # (batch_size, 1)
            
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

            decoder_inputs_embeds = self.embeddings[current_head_idx](next_token_id) # (batch_size, 1, d_model)
            past_key_values = decoder_outputs.past_key_values

        return generated_ids


if __name__ == '__main__':
    lm_heads_vocab_sizes = [10, 10, 10, 100]
    num_heads = len(lm_heads_vocab_sizes)
    special_vocab_size = 10
    model_dim = 128

    config = T5Config(
        vocab_size=max(lm_heads_vocab_sizes), 
        d_model=model_dim,
        d_kv=model_dim // 2,
        d_ff=model_dim * 2,
        num_layers=2,
        num_heads=4,
        decoder_start_token_id=-2,
        pad_token_id=-100
    )

    model = GenRec(
        config=config,
        lm_heads_vocab_sizes=lm_heads_vocab_sizes,
        special_vocab_size=special_vocab_size
    )


    input_ids = torch.tensor([[3, 5, 7, 1, 5, 4, -1, 1, 2, 3], [3, 6, 2, 5, 4, 2, -1, 1, 4, 2,]], dtype=torch.long)
    labels = torch.tensor([[1, 2, 3, 34, 3, 5, 7, 25], [1, 4, 2, 33, 3, 6, 2, 52]], dtype=torch.long)
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