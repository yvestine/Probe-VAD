"""Full YES/NO continuation sums with one shared visual encoding.
Visual preparation mirrors the original optimized engine; only the answer
scoring supports arbitrary token lengths. No modification to the original.
"""
import copy
import torch
from typing import List, Tuple
from src.video_score_utils import _move_processor_inputs

@torch.inference_mode()
def sequence_binary_likelihood(model, processor, conversations, threshold_batch_size=11):
    if not conversations or threshold_batch_size < 1:
        raise ValueError("Nonempty conversations and positive batch size required")
    tokenizer = processor.tokenizer
    # Load/decode the clip and run the image preprocessor exactly once.
    loaded_conversation = processor._load_multimodal_data(conversations[0])
    images = processor._gather_multimodal_data(loaded_conversation)
    if images is None:
        raise RuntimeError("threshold conversation contains no video")
    image_inputs = processor.process_images(images, return_tensors="pt")

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    image_inputs = _move_processor_inputs(image_inputs, device, dtype)

    # Build the ten text prompts against the same visual-token geometry without
    # loading or preprocessing the video again.
    text_inputs = []
    for conversation in conversations:
        current_conversation = copy.deepcopy(loaded_conversation)
        # Only the final threshold question differs. Preserve the already-loaded
        # frames, timestamps, and num_frames required by the chat template.
        source_texts = [
            content
            for message in conversation
            if isinstance(message.get("content"), (list, tuple))
            for content in message["content"]
            if isinstance(content, dict) and content.get("type") == "text"
        ]
        target_texts = [
            content
            for message in current_conversation
            if isinstance(message.get("content"), (list, tuple))
            for content in message["content"]
            if isinstance(content, dict) and content.get("type") == "text"
        ]
        if len(source_texts) != 1 or len(target_texts) != 1:
            raise RuntimeError("expected exactly one threshold text item")
        target_texts[0]["text"] = source_texts[0]["text"]
        values = processor._process_conversation_without_label(
            current_conversation,
            image_inputs,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        text_inputs.append(_move_processor_inputs(values, device, dtype))

    pixel_values = image_inputs["pixel_values"]
    grid_sizes = image_inputs["grid_sizes"]
    merge_sizes = image_inputs["merge_sizes"]
    modals = image_inputs["modals"]
    batched_num_patches = grid_sizes.prod(dim=1).div(merge_sizes ** 2).long()

    # This is the expensive visual path in VideoLLaMA3. Compute it once.
    mm_features = model.encode_images(pixel_values, grid_sizes, merge_sizes).to(device)
    mm_features = model._get_valid_visual_tokens(
        mm_features, batched_num_patches, modals
    )
    compression_mask = model._get_compression_mask(
        pixel_values, batched_num_patches, grid_sizes, merge_sizes, modals
    )

    prepared_inputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for values in text_inputs:
        input_ids = values["input_ids"].reshape(-1)
        attention_mask = values.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.reshape(-1)
        else:
            attention_mask = torch.ones_like(input_ids)

        current_features, current_mask = model._maybe_truncate_visual_tokens(
            mm_features,
            compression_mask,
            batched_num_patches,
            modals,
            input_ids,
            None,
        )
        if model.config.use_token_compression:
            (
                current_features,
                input_ids,
                attention_mask,
                _,
                _,
            ) = model._compress_visual_tokens(
                current_mask,
                current_features,
                input_ids,
                attention_mask,
                None,
                None,
            )

        inputs_embeds = model.get_model().embed_tokens(input_ids).clone()
        image_selected = input_ids == model.config.image_token_index
        inputs_embeds[image_selected] = (
            inputs_embeds[image_selected] * 0.0 + current_features
        )
        prepared_inputs.append((input_ids, attention_mask, inputs_embeds))

    # Teacher-force every token; append token embeddings AFTER visual compression.
    results = [{} for _ in conversations]
    for label, field in (("YES", "yes_loglik"), ("NO", "no_loglik")):
        ids = tokenizer(label, add_special_tokens=False).input_ids
        if not ids:
            raise ValueError("Empty continuation")
        candidate = torch.tensor(ids, device=device, dtype=torch.long)
        groups = {}
        for i, (_, _, embeddings) in enumerate(prepared_inputs):
            groups.setdefault(len(embeddings), []).append(i)
        for indices in groups.values():
            for offset in range(0, len(indices), threshold_batch_size):
                chunk = indices[offset:offset + threshold_batch_size]
                embeddings = torch.stack([torch.cat((prepared_inputs[i][2],
                    model.get_model().embed_tokens(candidate)), dim=0) for i in chunk])
                attention = torch.stack([torch.cat((prepared_inputs[i][1],
                    torch.ones_like(candidate)), dim=0) for i in chunk])
                outputs = model(inputs_embeds=embeddings, attention_mask=attention,
                    use_cache=False, return_dict=True)
                logits = outputs.logits[:, -len(ids)-1:-1, :].float()
                targets = candidate.expand(len(chunk), -1).unsqueeze(-1)
                sums = logits.log_softmax(-1).gather(2, targets).squeeze(-1).sum(-1)
                for i, value in zip(chunk, sums):
                    results[i][field] = float(value.item())
    probabilities = [float(torch.sigmoid(torch.tensor(d["yes_loglik"]-d["no_loglik"],
        dtype=torch.float64)).item()) for d in results]
    return probabilities, results
