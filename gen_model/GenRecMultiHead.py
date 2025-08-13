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
                 pad_token_id: int = 0,
                 config: T5Config = None):
        super().__init__(config)
        self.token_level_vocab_sizes = token_level_vocab_sizes
        self.special_vocab_size = special_vocab_size
        self.pad_token_id = pad_token_id

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
                head_labels[head_labels == self.pad_token_id] = -100
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

        # padding the logits to match the maximum vocab size for outputs
        max_vocab_size = max(self.token_level_vocab_sizes)
        batch_size = sequence_output.size(0)
        logits = torch.full(
            (batch_size, sequence_len, max_vocab_size),
            fill_value=-1e9,
            device=sequence_output.device,
            dtype=sequence_output.dtype
        )
        for idx, logit in enumerate(lm_logits):
            if logit is not None:
                vocab_size = logit.size(-1)
                logits[:, idx, :vocab_size] = logit

        if not return_dict:
            output = (logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
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
        max_length: int = 80,
        strategy: str = 'greedy',
        num_beams: int = 5,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ):
        """
        Generates token sequences autoregressively.

        Args:
            input_ids (`torch.Tensor`): Input sequence IDs.
            attention_mask (`torch.Tensor`): Attention mask.
            max_length (`int`): Maximum length of the generated sequence.
            strategy (`str`): 'greedy', 'sampling', or 'beam_search'.
            num_beams (`int`): Number of beams for beam search.
            temperature (`float`): Temperature for sampling.
            top_k (`int`): Top-k filtering for sampling.
            top_p (`float`): Top-p (nucleus) filtering for sampling.

        Returns:
            `torch.Tensor`: The generated sequence of token IDs.
        """
        self.eval()
        if strategy == 'greedy':
            generated_ids = self._greedy_search(input_ids, attention_mask, max_length)
        elif strategy == 'sampling':
            generated_ids = self._sampling_search(input_ids, attention_mask, max_length, temperature, top_k, top_p)
        elif strategy == 'beam_search':
            generated_ids = self._beam_search(input_ids, attention_mask, max_length, num_beams)
        else:
            raise ValueError(f"Unknown generation strategy: {strategy}")

        batch_size = input_ids.shape[0]
        input_length = (input_ids != self.pad_token_id).sum(dim=-1)
        generated_length = input_length - len(self.level_offsets)  # max generated length is all candidate tokens
        indices = torch.arange(max_length, device=generated_ids.device)
        mask = indices.expand(batch_size, -1) >= generated_length.unsqueeze(-1)
        generated_ids = generated_ids.masked_fill(mask, self.pad_token_id)

        return generated_ids

    def _greedy_search(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 20
    ):
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

    def _sampling_search(self, input_ids, attention_mask, max_length, temperature, top_k, top_p):
        self.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        inputs_embeds = self.embeddings(input_ids)
        encoder_outputs = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state

        decoder_start_ids = torch.full(
            (batch_size, 1), self.config.decoder_start_token_id, dtype=torch.long, device=device
        )
        decoder_inputs_embeds = self.embeddings(decoder_start_ids)

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
            
            last_hidden_state = decoder_outputs.last_hidden_state
            lm_head = self.lm_heads[token_level_idx]
            logits = lm_head(last_hidden_state)
            next_token_logits = logits.squeeze(1)

            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            if top_k > 0:
                top_k_logits, _ = torch.topk(next_token_logits, top_k)
                k_th_value = top_k_logits[:, -1]
                indices_to_remove = next_token_logits < k_th_value[:, None]
                next_token_logits[indices_to_remove] = -float('Inf')

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('Inf')

            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            next_token_id = self.level_offsets[token_level_idx] + next_token_id
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            decoder_inputs_embeds = self.embeddings(next_token_id)
            past_key_values = decoder_outputs.past_key_values

        return generated_ids

    def _beam_search(self, input_ids, attention_mask, max_length, num_beams):
        self.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device
        token_levels = len(self.token_level_vocab_sizes)

        inputs_embeds = self.embeddings(input_ids)
        encoder_outputs = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state

        expanded_batch_size = batch_size * num_beams
        expanded_encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_beams, dim=0)
        expanded_attention_mask = attention_mask.repeat_interleave(num_beams, dim=0)

        decoder_start_ids = torch.full(
            (expanded_batch_size, 1), self.config.decoder_start_token_id, dtype=torch.long, device=device
        )
        
        generated_ids = decoder_start_ids

        beam_scores = torch.zeros((batch_size, num_beams), device=device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        past_key_values = None

        for step in range(max_length):
            token_level_idx = step % token_levels

            current_input_ids = generated_ids[:, -1].unsqueeze(-1) # (batch_size * num_beams, 1)
            decoder_inputs_embeds = self.embeddings(current_input_ids)

            decoder_outputs = self.decoder(
                inputs_embeds=decoder_inputs_embeds,
                encoder_hidden_states=expanded_encoder_hidden_states,
                encoder_attention_mask=expanded_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

            last_hidden_state = decoder_outputs.last_hidden_state 
            past_key_values = decoder_outputs.past_key_values
            
            lm_head = self.lm_heads[token_level_idx]
            logits = lm_head(last_hidden_state)
            next_token_logits = logits.squeeze(1) # (batch_size * num_beams, vocab_size)

            log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)

            scores = log_probs + beam_scores.unsqueeze(1) # (batch_size * num_beams, vocab_size)

            vocab_size = self.token_level_vocab_sizes[token_level_idx]
            scores = scores.view(batch_size, num_beams * vocab_size)

            next_beam_scores, next_beam_indices = torch.topk(scores, num_beams, dim=1, largest=True, sorted=True)

            next_beam_ids = torch.div(next_beam_indices, vocab_size, rounding_mode='floor')
            next_token_ids = next_beam_indices % vocab_size

            batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).repeat(1, num_beams)
            beam_idx = (batch_idx * num_beams + next_beam_ids).view(-1)
            
            past_key_values = self._reorder_cache(past_key_values, beam_idx)
            
            next_token_ids_with_offset = self.level_offsets[token_level_idx] + next_token_ids
            generated_ids = generated_ids[beam_idx]
            generated_ids = torch.cat([generated_ids, next_token_ids_with_offset.view(-1, 1)], dim=-1)

            beam_scores = next_beam_scores.view(-1)

        best_sequences = generated_ids.view(batch_size, num_beams, -1)[:, 0, :]
        
        return best_sequences[:, 1:]

    def _reorder_cache(self, past, beam_idx):
        reordered_past = []
        for layer_past in past:
            reordered_past.append(tuple(past_state.index_select(0, beam_idx) for past_state in layer_past))
        return tuple(reordered_past)


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
        print("\nGreedy search:")
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=8
        )
        print(f"Generated IDs: {generated_ids}")

        print("\nSampling search:")
        generated_ids_sampling = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=8,
            strategy='sampling',
            temperature=0.7,
            top_k=50
        )
        print(f"Generated IDs (sampling): {generated_ids_sampling}")

        print("\nBeam search:")
        generated_ids_beam = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=8,
            strategy='beam_search',
            num_beams=4
        )
        print(f"Generated IDs (beam search): {generated_ids_beam}")