# 🔍 Phân Tích Chi Tiết: Source Code HRPO Thay Đổi Gì So Với Thư Viện Gốc?

Dưới đây là **tất cả** các thay đổi HRPO thực hiện trên 3 thư viện gốc: `transformers`, `trl`, và `unsloth`.

---

## 1. `transformers/models/qwen2/modeling_qwen2.py` — Thêm Gating Mechanism

> Thay đổi so với `transformers` gốc (v4.48+)

### 1a. Thêm class `ThinkingResidualLambda` (dòng 469–489)
**🆕 Hoàn toàn mới** — không tồn tại trong thư viện gốc.

```python
class ThinkingResidualLambda(nn.Module):
    c = 8.0
    def __init__(self, config):
        self.Lambda = nn.Parameter(torch.randn(config.hidden_size))
    
    def reset_lambda_parameters(self, r_min=0.9, r_max=0.999):
        nn.init.uniform_(self.Lambda, a=r_min, b=r_max)
        self.Lambda.data.copy_(- torch.log((self.Lambda ** (-1./self.c)) - 1))
    
    def forward(self, r_t):
        a_t = exp(-c * softplus(-Λ) * r_t)
        return a_t
```

**Paper reference**: Eq. 4 — `a_t` controls the blend ratio. Initialized via `r_min` (Appendix A).

### 1b. Thêm 3 thuộc tính mới vào `Qwen2Model.__init__()` (dòng 517–519)

```diff
 class Qwen2Model(Qwen2PreTrainedModel):
     def __init__(self, config):
         ...
         self.rotary_emb = Qwen2RotaryEmbedding(config=config)
+        self.thinking_residual_gate_r = nn.Linear(config.hidden_size, config.hidden_size)
+        self.thinking_residual_gate_i = nn.Linear(config.hidden_size, config.hidden_size)
+        self.thinking_residual_Lambda = ThinkingResidualLambda(config)
         self.post_init()
```

### 1c. Thêm method `thinking_residual()` vào `Qwen2Model` (dòng 530–534)

```diff
+    def thinking_residual(self, embeds, residual, eps=1e-8):
+        r_t = sigmoid(self.thinking_residual_gate_r(embeds))
+        i_t = sigmoid(self.thinking_residual_gate_i(embeds))
+        a_t = self.thinking_residual_Lambda(r_t)
+        return a_t * embeds + sqrt(1 - a_t² + eps) * (i_t * residual), a_t
```

**Paper reference**: Eq. 4 — `e_{t+1} = a_t · ê_{t+1} + √(1-a_t²) · (i_t · h_{t+1})`

---

## 2. `transformers/models/llama/modeling_llama.py` — Tương Tự Qwen2

Cùng 3 thay đổi tương tự (ThinkingResidualLambda, gates, thinking_residual method) được áp dụng cho `LlamaModel` (dòng 539–555).

---

## 3. `transformers/generation/utils.py` — Sửa Generation Loop

> Thay đổi so với `transformers` gốc

### 3a. Thêm tham số `return_thinking_embeds` và `is_inference` vào `generate()` (dòng 1988–1989)

```diff
 def generate(
     self, inputs=None, generation_config=None, ...,
+    return_thinking_embeds: bool = False,
+    is_inference: bool = False,
     **kwargs,
 ):
```

### 3b. Truyền `return_thinking_embeds` và `is_inference` xuống `_sample()` (dòng 2337–2338)

```diff
 result = self._sample(
     ...,
+    return_thinking_embeds=return_thinking_embeds,
+    is_inference=is_inference,
     **model_kwargs,
 )
```

### 3c. Sửa đổi lớn trong `_sample()` (dòng 3198–3427)

**Tham số mới** (dòng 3207–3208):
```diff
-def _sample(self, input_ids, logits_processor, stopping_criteria, generation_config, synced_gpus, streamer, **model_kwargs):
+def _sample(self, ..., processing_class=None, return_thinking_embeds=False, is_inference=False, **model_kwargs):
```

**Khởi tạo buffers** (dòng 3287–3294):
```diff
+is_thinking, last_thinking_states = None, None
+thinking_embeds = [self.get_input_embeddings()(input_ids)] if return_thinking_embeds else []
+thinking_mask = [torch.zeros_like(input_ids, dtype=torch.bool)] if return_thinking_embeds else []
+embeds_ratio = [torch.ones_like(input_ids, dtype=torch.float32)] if return_thinking_embeds else []
```

**Truyền is_thinking/last_thinking_states vào model** (dòng 3303–3305):
```diff
+# prepare is_thinking and last_thinking_states for latent reasoning
+model_inputs.update({"is_thinking": is_thinking} if is_thinking is not None else {})
+model_inputs.update({"last_thinking_states": last_thinking_states} if last_thinking_states is not None else {})
```

**Greedy decoding cho inference** (dòng 3352–3356):
```diff
 probs = F.softmax(next_token_scores, dim=-1)
+if is_inference:
+    next_tokens = torch.argmax(probs, dim=-1)
+else:
     next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
```

**Tính `last_thinking_states` — Paper Eq. 3** (dòng 3367–3372):
```diff
+strs = processing_class.batch_decode(input_ids[:, input_len:])
+is_thinking = [self.answer_start not in s for s in strs]
+last_thinking_states = torch.einsum('bv,vd->bd', probs, self.get_input_embeddings().weight)
+last_thinking_states /= torch.sqrt((probs ** 2).sum(-1, keepdim=True)).to(last_thinking_states.dtype)
```

> [!IMPORTANT]
> **Đây là Eq. 3 trong paper**: `h_{t+1} = Σ(p_i · e_i) / ‖p‖₂` — projected hidden state qua weighted interpolation over vocabulary embeddings, normalized.

**Thu thập thinking data cho training** (dòng 3374–3381):
```diff
+if return_thinking_embeds and outputs.hidden_states is not None:
+    thinking_embeds.append(outputs.hidden_states[0].unsqueeze(1))
+    thinking_mask.append(torch.tensor(outputs.hidden_states[1], ...))
+    embeds_ratio.append(torch.tensor(outputs.hidden_states[2], ...))
```

> **Hack thú vị**: `hidden_states` field được "hijack" để truyền `[thinking_embeds, is_thinking, embeds_ratio]` thay vì standard hidden states trong unsloth fast forward path.

**Return thinking data** (dòng 3417–3425):
```diff
+if return_thinking_embeds:
+    return input_ids, torch.cat(thinking_embeds, dim=1), 
+           torch.cat(thinking_mask, dim=1), torch.cat(embeds_ratio, dim=1)
```

---

## 4. `trl/trainer/grpo_trainer.py` — Sửa GRPO Trainer

### 4a. `_get_per_token_logps()` — Thêm tham số `inputs_embeds` (dòng 478)

```diff
-def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
+def _get_per_token_logps(self, model, input_ids, inputs_embeds, attention_mask, logits_to_keep):
     logits = model(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, ...).logits
```

### 4b. `_prepare_inputs()` — Generation trả về thinking data (dòng 564–569)

```diff
-prompt_completion_ids = unwrapped_model.generate(prompt_ids, ...)
+prompt_completion_ids, thinking_embeds, thinking_mask, embeds_ratio = unwrapped_model.generate(
+    prompt_ids, ..., return_thinking_embeds=True,
+)
```

### 4c. `_prepare_inputs()` — Reference model KHÔNG dùng thinking_embeds (dòng 590–597)

```python
# Reference model sử dụng inputs_embeds=None → chỉ dùng token IDs
ref_per_token_logps = self._get_per_token_logps(
    self.ref_model, prompt_completion_ids, None, attention_mask, logits_to_keep
)
```

> [!IMPORTANT]
> **Paper Section 3.2**: "For KL divergence, we compute log probabilities using solely token IDs for π_ref" — đảm bảo training stability bằng cách chỉ dùng discrete tokens cho reference model.

### 4d. `_prepare_inputs()` — Return thinking data (dòng 686–696)

```diff
 return {
     "prompt_ids": prompt_ids, "prompt_mask": prompt_mask,
     "completion_ids": completion_ids, "completion_mask": completion_mask,
+    "thinking_embeds": thinking_embeds,
+    "thinking_mask": thinking_mask,
+    "embeds_ratio": embeds_ratio,
     "ref_per_token_logps": ref_per_token_logps,
     "advantages": advantages,
 }
```

---

## 5. `unsloth/models/llama.py` — Fast Forward + Thinking Residual

### 5a. Training Path: `LlamaModel_fast_forward()` (dòng 606–669)

```diff
 # Khi cả input_ids và inputs_embeds đều có → inputs_embeds là thinking_embeds
+if input_ids is not None and inputs_embeds is not None:
+    thinking_embeds = inputs_embeds
     # (Không raise error như gốc)
 
 # Embed positions
 inputs_embeds = self.embed_tokens(input_ids)

+thinking_mask = kwargs.get('thinking_mask')
+if thinking_mask is not None:
+    new_inputs_embeds = inputs_embeds.clone()
+    new_inputs_embeds[thinking_mask] = self.thinking_residual(
+        inputs_embeds[thinking_mask], thinking_embeds[thinking_mask],
+    )[0].to(inputs_embeds.dtype)
+    inputs_embeds = new_inputs_embeds
```

### 5b. Inference Path: `LlamaModel_fast_forward_inference()` (dòng 940–1027)

```diff
+is_thinking = kwargs.get('is_thinking')
+last_thinking_states = kwargs.get('last_thinking_states')
+if is_thinking is not None and last_thinking_states is not None:
+    thinking_embeds = last_thinking_states
+    X_hat, a_t = self.model.thinking_residual(X, last_thinking_states.unsqueeze(1))
+    embeds_ratio = a_t.mean(-1).squeeze()
+    embeds_ratio[~torch.tensor(is_thinking)] = 1.
+    X[is_thinking] = X_hat[is_thinking].to(X.dtype)

 # Return value: hijack hidden_states để truyền thinking data
 return BaseModelOutputWithPast(
     last_hidden_state = X,
     past_key_values = next_decoder_cache,
+    hidden_states = [] if is_thinking is None else [thinking_embeds, is_thinking, embeds_ratio],
 )
```

### 5c. CausalLM Forward: Return hidden_states thay vì logits (dòng 1098–1107)

```diff
+# Env var UNSLOTH_RETURN_HIDDEN_STATES=1 → return hidden_states thay vì tính logits
+if self.training and os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1":
+    return CausalLMOutputWithPast(
+        logits = hidden_states,  # ← hidden states được đặt vào trường logits!
+        ...
+    )
```

> **Memory optimization**: Bỏ qua bước nhân lm_head (V×H matrix), tiết kiệm rất nhiều VRAM.

---

## 6. `unsloth/models/rl_replacements.py` — **THAY ĐỔI LỚN NHẤT** ⚡

> File này **viết lại hoàn toàn** `compute_loss` và `_get_per_token_logps` của GRPO trainer.

### 6a. `_get_per_token_logps` bị viết lại (dòng 210–233)

```diff
-# Gốc: Tính logprobs từ logits
-logits = model(input_ids=...).logits
-return selective_log_softmax(logits, input_ids)

+# HRPO: Return None → trigger efficient path thay thế
+def _get_per_token_logps(self, model, input_ids, inputs_embeds, attention_mask, logits_to_keep):
+    if os.environ.get('UNSLOTH_USE_NEW_MODEL', '0') == '0':
+        return None  # → Trigger grpo_accumulated_loss (efficient path)
```

### 6b. `grpo_accumulated_loss()` — Efficient GRPO with Thinking (dòng 380–431)

```python
def grpo_accumulated_loss(trainer, input_ids, thinking_embeds, thinking_mask, ...):
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"  # Bật return hidden states
    
    # 1. Reference model: CHỈ dùng input_ids (KHÔNG dùng thinking_embeds)
    with disable_adapter():
        old_hidden_states = trainer.model(input_ids=input_ids, ...).logits  # ← thực ra là hidden_states
    
    # 2. Policy model: DÙng cả thinking_embeds + thinking_mask
    new_hidden_states = trainer.model(
        input_ids=input_ids, 
        inputs_embeds=thinking_embeds,    # ← thinking embeddings
        thinking_mask=thinking_mask,       # ← mask chỉ vùng thinking
        ...
    ).logits  # ← thực ra là hidden_states (nhờ UNSLOTH_RETURN_HIDDEN_STATES=1)
    
    # 3. Tính loss bằng UnslothEfficientGRPO
    loss, completion_length, mean_kl = UnslothEfficientGRPO.apply(
        new_hidden_states, old_hidden_states, lm_head,
        completion_input_ids, completion_mask, advantages, beta, ...
    )
```

> [!IMPORTANT]
> **Điểm mấu chốt theo paper**: Reference model dùng pure token IDs, policy model dùng hybrid (token + hidden states). Đây là cách HRPO tính KL divergence: `π_ref` chỉ thấy tokens, còn `π_θ` thấy hybrid.

### 6c. `UnslothEfficientGRPO` — Custom Autograd Function (dòng 289–376)

Tối ưu bộ nhớ bằng cách:
- **Chunk** batch thành nhiều phần nhỏ
- Dùng `torch.func.grad_and_value` → tính gradient + loss cùng lúc
- Dùng `torch.compile` cho mỗi chunk
- Không lưu activations intermediate → tiết kiệm VRAM đáng kể

### 6d. `compute_loss` viết lại hoàn toàn (dòng 446–508)

```diff
-# Gốc: Tính per_token_logps rồi loss
-per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, ...)
-per_token_kl = exp(ref - new) - (ref - new) - 1
-loss = ...

+# HRPO: Truyền thinking_embeds vào model
+thinking_embeds, thinking_mask = inputs["thinking_embeds"], inputs["thinking_mask"]
+per_token_logps = self._get_per_token_logps(model, input_ids, thinking_embeds, attention_mask, ...)
+
+if per_token_logps is None:
+    # Efficient path: tính loss trực tiếp trên hidden states
+    loss, completion_length, mean_kl = grpo_accumulated_loss(
+        self, input_ids, thinking_embeds, thinking_mask, ...
+    )
+
+# Log HRPO-specific metrics
+embeds_ratio = inputs["embeds_ratio"]
+self._metrics["embeds_ratio"].append(...)   # Tỷ lệ token embedding
+self._metrics["hidden_ratio"].append(...)   # Tỷ lệ hidden state
```

---

## 7. `patch.py` — Optimizer Parameter Groups

> Đã covered trong report trước, tóm tắt: tách 4 nhóm LR cho LoRA params, gate params, Lambda params.

---

## 8. `unsloth/trainer.py` — Trainer Compatibility

> Đã covered: `UnslothTrainer` override `create_optimizer` cho embedding LR.

---

## Tổng Kết Tất Cả Thay Đổi

| # | File | Thay đổi | Mục đích |
|---|---|---|---|
| 1 | `transformers/models/qwen2/modeling_qwen2.py` | Thêm `ThinkingResidualLambda`, gates, `thinking_residual()` | Core gating mechanism (Eq. 4) |
| 2 | `transformers/models/llama/modeling_llama.py` | Tương tự #1 | Hỗ trợ Llama models |
| 3 | `transformers/generation/utils.py` | Sửa `generate()` + `_sample()` | Latent reasoning trong generation loop (Eq. 3) |
| 4 | `trl/trainer/grpo_trainer.py` | Sửa `_prepare_inputs()`, `_get_per_token_logps()` | Truyền thinking data qua training pipeline |
| 5 | `unsloth/models/llama.py` (training) | Sửa `LlamaModel_fast_forward()` | Apply gating khi training |
| 6 | `unsloth/models/llama.py` (inference) | Sửa `LlamaModel_fast_forward_inference()` | Apply gating + return thinking state khi inference |
| 7 | `unsloth/models/llama.py` (CausalLM) | Thêm `UNSLOTH_RETURN_HIDDEN_STATES` | Trả hidden states thay vì logits → tiết kiệm VRAM |
| 8 | **`unsloth/models/rl_replacements.py`** | **Viết lại** `compute_loss`, `_get_per_token_logps`, thêm `UnslothEfficientGRPO`, `grpo_accumulated_loss` | **Core HRPO training**: ref model dùng tokens, policy model dùng hybrid; memory-efficient chunked loss |
| 9 | `patch.py` | Patch optimizer | 4 LR groups cho HRPO params |
| 10 | `unsloth/trainer.py` | Override `create_optimizer` | Embedding LR riêng |

> [!CAUTION]
> **File quan trọng nhất mà báo cáo trước chưa cover**: `unsloth/models/rl_replacements.py` — chứa toàn bộ logic tính loss cho HRPO training, bao gồm cách reference model và policy model được forward khác nhau.
