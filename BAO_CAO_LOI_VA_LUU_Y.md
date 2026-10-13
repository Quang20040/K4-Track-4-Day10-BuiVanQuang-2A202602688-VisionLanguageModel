# BÁO CÁO LỖI, PHÂN TÍCH NGUYÊN NHÂN VÀ HƯỚNG DẪN TỐI ƯU HÓA (TROUBLESHOOTING GUIDE)
## VinUni AI Track 4 Day 10 — Finetune Qwen3-VL-4B bằng QLoRA trên Google Colab T4

**Học viên:** Bùi Văn Quang  
**MSSV:** 2A202602688  
**Môi trường:** Google Colab GPU NVIDIA Tesla T4 (16GB VRAM)  
**Mô hình:** `Qwen/Qwen3-VL-4B-Instruct` (Tổng tham số: ~4.44B)  

---

### MỤC LỤC
1. [Quản lý tài nguyên VRAM và phòng chống CUDA Out-of-Memory (OOM)](#1-quản-lý-tài-nguyên-vram-và-phòng-chống-cuda-out-of-memory-oom)
2. [Lượng tử hóa 4-bit NF4 và cơ chế skip ViT / lm_head](#2-lượng-tử-hóa-4-bit-nf4-và-cơ-chế-skip-vit--lm_head)
3. [Cấu hình PEFT LoRA target modules và bẫy sai lệch số lượng module](#3-cấu-hình-peft-lora-target-modules-và-bẫy-sai-lệch-số-lượng-module)
4. [Huấn luyện Projector (E2, E3) và lỗi quên lưu / nạp `projector_weights.pt`](#4-huấn-luyện-projector-e2-e3-và-lỗi-quên-lưu--nạp-projector_weightspt)
5. [Cơ chế Autoregressive Loss Masking (-100) và Collation đa phương thức](#5-cơ-chế-autoregressive-loss-masking--100-và-collation-đa-phương-thức)
6. [Hệ tọa độ chuẩn hóa 0–1000 và bẫy lệch thang đo Pixel](#6-hệ-tọa-độ-chuẩn-hóa-01000-và-bẫy-lệch-thang-đo-pixel)
7. [Chống rò rỉ dữ liệu (Data Leakage) bằng Session-based Splitting](#7-chống-rò-rỉ-dữ-liệu-data-leakage-bằng-session-based-splitting)
8. [Phân tích khoảng tin cậy Bootstrap CI95 trên tập Test 311 ảnh](#8-phân-tích-khoảng-tin-cậy-bootstrap-ci95-trên-tập-test-311-ảnh)
9. [Cơ chế Verifiable Reward cho GRPO và phòng chống Reward Hacking](#9-cơ-chế-verifiable-reward-cho-grpo-và-phòng-chống-reward-hacking)
10. [Chiến lược phòng chống mất phiên làm việc (Colab Disconnection Resilience)](#10-chiến-lược-phòng-chống-mất-phiên-làm-việc-colab-disconnection-resilience)

---

### 1. Quản lý tài nguyên VRAM và phòng chống CUDA Out-of-Memory (OOM)

#### Hiện tượng:
GPU Tesla T4 trên Colab chỉ có tối đa 15.84 GB VRAM khả dụng. Khi fine-tune mô hình đa phương thức 4.4B tham số:
- Activation của chuỗi token văn bản kết hợp 220 visual tokens mỗi ảnh.
- Gradient và optimizer states (AdamW tiêu tốn 8 bytes cho mỗi tham số trainable: 4 bytes cho first moment $m_t$, 4 bytes cho second moment $v_t$).
- Nếu cấu hình batch size lớn (ví dụ batch size = 4 hoặc 8), GPU sẽ lập tức ném ngoại lệ `torch.cuda.OutOfMemoryError`.

#### Phân tích nguyên nhân gốc:
- Forward pass của Vision Transformer và LLM tạo ra các intermediate activation maps rất lớn.
- Nếu không bật **Gradient Checkpointing**, toàn bộ activation của tất cả 36 transformer layers trong LLM và 24 blocks trong ViT được lưu giữ trên VRAM để tính backward, khiến peak VRAM vượt ngưỡng 16GB.

#### Giải pháp tối ưu:
1. **Gradient Checkpointing:**
   ```python
   model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
   model.config.use_cache = False  # Bắt buộc tắt KV cache khi training
   ```
   Gradient Checkpointing giải phóng intermediate activations trong forward pass và tái tính toán (recompute) trong backward pass, giúp giảm VRAM từ ~22GB xuống còn ~8–12GB.
2. **Effective Batch Size bằng Gradient Accumulation:**
   - Cấu hình `batch_size = 2`, `grad_accum = 4` $\rightarrow$ Effective batch size = $2 \times 4 = 8$.
   - Nếu gặp cảnh báo bộ nhớ đỉnh (ví dụ ở E3 khi có thêm LoRA trên ViT), có thể chuyển linh hoạt sang `batch_size = 1`, `grad_accum = 8` để giữ nguyên effective batch size = 8 mà giảm một nửa activation memory cho mỗi micro-step.
3. **Thu hồi bộ nhớ GPU giữa các Experiment:**
   ```python
   del model, optimizer, scheduler
   import gc
   gc.collect()
   torch.cuda.empty_cache()
   torch.cuda.reset_peak_memory_stats()
   ```

---

### 2. Lượng tử hóa 4-bit NF4 và cơ chế skip ViT / lm_head

#### Hiện tượng:
Nếu cấu hình lượng tử hóa mặc định toàn bộ mô hình:
```python
BitsAndBytesConfig(load_in_4bit=True)
```
thì ViT và khối Projector (`visual.merger`, `visual.deepstack_merger_list`) sẽ bị nén sang kiểu dữ liệu `uint8` 4-bit (NF4). Khi chuyển sang E2/E3 cần mở unfreeze projector để huấn luyện gradient, PyTorch sẽ báo lỗi hoặc trọng số bị đóng băng do không thể tính đạo hàm trên trọng số đã bị lượng tử hóa.

#### Phân tích nguyên nhân gốc:
- Khối Projector trong Qwen3-VL có nhiệm vụ ánh xạ không gian biểu diễn từ Vision Transformer sang Language Model space.
- Các module lượng tử hóa `Linear4bit` không lưu giữ tensor gradient chuẩn mà lưu trữ các bảng tra lượng tử (quant state). Trọng số 4-bit không có float gradient trực tiếp.
- `lm_head` nếu bị lượng tử hóa cũng gây suy giảm nghiêm trọng độ chính xác phân phối xác suất token (perplexity spike).

#### Giải pháp:
Sử dụng cờ `llm_int8_skip_modules` trong cấu hình BitsAndBytesConfig:
```python
quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=["visual", "lm_head"]
)
```
- Khối `visual` (gồm ViT và projector) cùng `lm_head` được giữ nguyên ở định dạng `fp16`.
- Chỉ các lớp tuyến tính của `language_model` được lượng tử hóa 4-bit NF4, tiết kiệm được ~7GB VRAM mà vẫn đảm bảo ViT và Projector có thể gắn LoRA hoặc unfreeze full gradient ở các setup sau.

---

### 3. Cấu hình PEFT LoRA target modules và bẫy sai lệch số lượng module

#### Hiện tượng:
- Một lỗi thường gặp là sử dụng target modules quá rộng, ví dụ: `target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]`.
- Hoặc ngược lại, viết regex không khớp với tiền tố cây module thực tế của `Qwen3-VL` (ví dụ thiếu tiền tố `model.language_model`).

#### Phân tích cấu trúc module thực tế của `Qwen/Qwen3-VL-4B-Instruct`:
- **LLM:** Có 36 decoder layers. Mỗi layer gồm:
  - Self-attention: `q_proj`, `k_proj`, `v_proj`, `o_proj` (4 modules)
  - MLP: `gate_proj`, `up_proj`, `down_proj` (3 modules)
  - $\rightarrow 36 \times 7 = \mathbf{252}$ LoRA modules.
- **ViT:** Có 24 blocks. Mỗi block gồm:
  - Attention: `attn.qkv`, `attn.proj` (2 modules)
  - MLP: `mlp.linear_fc1`, `mlp.linear_fc2` (2 modules)
  - $\rightarrow 24 \times 4 = \mathbf{96}$ LoRA modules.
- **Tổng LoRA modules cho E3:** $252 + 96 = \mathbf{348}$ modules!

#### Giải pháp chuẩn:
Sử dụng regex chuẩn hóa:
```python
# Cho E1 và E2:
llm_regex = r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"

# Cho E3 (bổ sung ViT):
vit_regex = r".*visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(linear_fc1|linear_fc2))"
e3_regex = f"({llm_regex}|{vit_regex})"
```
Kiểm tra thực tế bằng assertion:
```python
matched = [n for n, m in peft_model.named_modules() if hasattr(m, "lora_A")]
if setup == "E1" or setup == "E2":
    assert len(matched) == 252, f"Kỳ vọng 252 LoRA modules, thực tế: {len(matched)}"
elif setup == "E3":
    assert len(matched) == 348, f"Kỳ vọng 348 LoRA modules, thực tế: {len(matched)}"
```

---

### 4. Huấn luyện Projector (E2, E3) và lỗi quên lưu / nạp `projector_weights.pt`

#### Triệu chứng kinh điển:
Khi đánh giá mô hình, chỉ số **F1 của E2 và E3 bằng đúng F1 của E1** (hoặc chỉ khác nhau ở sai số ngẫu nhiên rất nhỏ).

#### Nguyên nhân cốt lõi:
Hàm `peft_model.save_pretrained(checkpoint_dir)` của thư viện PEFT **chỉ lưu các trọng số LoRA adapter** (`adapter_model.safetensors`). Khối Projector (`visual.merger` và `visual.deepstack_merger_list`) trong E2 và E3 là các tham số mở toàn phần (full fine-tuned parameters), **hoàn toàn không phải là LoRA adapter**!
Do đó, khi gọi `save_pretrained()`, các trọng số Projector đã được huấn luyện biến mất. Khi nạp lại mô hình để inference, code nạp base model gốc và nạp adapter của E2/E3, dẫn đến Projector vẫn là trọng số gốc chưa huấn luyện $\rightarrow$ Kết quả trở về tương đương E1!

#### Giải pháp dứt điểm:
1. **Khi lưu checkpoint:** Tách riêng các tham số projector có `requires_grad=True` và lưu vào file riêng:
   ```python
   extra_trainable = {
       name: param.detach().cpu()
       for name, param in model.named_parameters()
       if ("visual.merger" in name or "visual.deepstack_merger_list" in name) and param.requires_grad
   }
   if extra_trainable:
       torch.save(extra_trainable, os.path.join(checkpoint_dir, "projector_weights.pt"))
   ```
2. **Khi nạp lại để suy luận:**
   ```python
   proj_file = os.path.join(checkpoint_dir, "projector_weights.pt")
   if os.path.exists(proj_file):
       extra = torch.load(proj_file, map_location="cpu")
       state_dict = model.state_dict()
       for k, v in extra.items():
           if k in state_dict:
               state_dict[k].copy_(v.to(state_dict[k].device, dtype=state_dict[k].dtype))
   ```
3. **Đưa Projector về Float32 khi huấn luyện:**
   Trọng số Projector gốc ở `fp16`. Khi huấn luyện full tham số trong môi trường mixed precision, gradient tích lũy trong float16 rất dễ bị underflow (gradient triệt tiêu về 0) hoặc overflow (tràn số thành NaN). Chuyển Projector sang `float32` giải quyết triệt để tính ổn định số học.

---

### 5. Cơ chế Autoregressive Loss Masking (-100) và Collation đa phương thức

#### Hiện tượng & rủi ro:
Nếu tính Cross-Entropy loss trên toàn bộ chuỗi token (bao gồm cả prompt hệ thống và 220 visual tokens của ảnh):
- Mô hình sẽ bị phân tán gradient để học "dự đoán lại" các đặc trưng ảnh hoặc học lại câu lệnh hướng dẫn (prompt).
- Khả năng tuân thủ định dạng JSON grounding bị suy giảm nghiêm trọng.

#### Bản chất kỹ thuật của Loss Masking:
Trong hàm Cross-Entropy Loss của PyTorch, tham số `ignore_index = -100` loại trừ hoàn toàn các vị trí có nhãn bằng `-100` ra khỏi việc tính loss và gradient.
```python
labels = full_ids.clone()
prefix_len = prompt_ids.shape[1]
labels[:, :prefix_len] = -100  # Mask toàn bộ prompt và image tokens
```
Khi đó, chỉ các token thuộc phần phản hồi của assistant (bắt đầu bằng `[` và kết thúc bằng `]` kèm token kết thúc `<|im_end|>`) tham gia vào hàm mất mát.

#### Quy tắc Collation đa phương thức:
- **Dữ liệu văn bản (`input_ids`, `attention_mask`, `labels`):** Cần đệm (pad) về cùng độ dài chuỗi dài nhất trong batch. `labels` phải đệm bằng `-100`, `input_ids` đệm bằng `pad_token_id`, `attention_mask` đệm bằng `0`.
- **Dữ liệu hình ảnh (`pixel_values`, `image_grid_thw`):** **TUYỆT ĐỐI KHÔNG PADDING**. Phải ghép nối (concatenate) dọc theo chiều thứ 0 (`dim=0`), bởi vì cơ chế xử lý Vision Transformer động của Qwen3-VL quản lý token ảnh thông qua bảng tọa độ lưới không-thời gian `image_grid_thw`.

---

### 6. Hệ tọa độ chuẩn hóa 0–1000 và bẫy lệch thang đo Pixel

#### Triệu chứng:
Mô hình xuất ra JSON hoàn toàn hợp lệ, nhưng khi chạy `scripts/score.py` thì **F1 = 0.000**.

#### Phân tích nguyên nhân:
- Kích thước ảnh thực tế là $640 \times 360$ pixels.
- Quy ước Grounding của Qwen3-VL sử dụng hệ tọa độ tương đối chuẩn hóa về đoạn $[0, 1000]$:
  $$x_{\text{norm}} = \frac{x_{\text{pixel}}}{W} \times 1000, \quad y_{\text{norm}} = \frac{y_{\text{pixel}}}{H} \times 1000$$
- Nếu model xuất ra tọa độ pixel tuyệt đối (ví dụ $x \in [0, 640], y \in [0, 360]$) nhưng hàm chấm điểm so khớp với ground truth ở thang $0–1000$, thì IoU giữa box dự đoán và ground truth sẽ luôn nhỏ hơn 0.5 (thậm chí xấp xỉ 0) $\rightarrow$ Không có cặp nào match $\rightarrow \text{TP} = 0, \text{F1} = 0$.

#### Giải pháp kiểm chứng:
Unit test `test_pixel_vs_norm1000_scale_trap` trong `tests/test_score.py` đã chỉ rõ điều này. Trong toàn bộ pipeline huấn luyện và suy luận:
- Luôn chuyển YOLO $[x_c, y_c, w, h]$ sang thang $0–1000$ bằng `yolo_to_box()`.
- Luôn kiểm tra điều kiện nghiêm ngặt: $x_1 < x_2$ và $y_1 < y_2$.

---

### 7. Chống rò rỉ dữ liệu (Data Leakage) bằng Session-based Splitting

#### Phân tích rủi ro trong dữ liệu Dashcam:
- Dữ liệu Road Damage là các khung hình (frames) được trích xuất từ video camera hành trình xe chạy liên tục trên đường với tốc độ 24–30 khung hình/giây.
- Hai khung hình cách nhau 0.5 giây có mặt đường, góc nhìn, ổ gà và điều kiện ánh sáng giống nhau đến 99%.
- Nếu thực hiện **Random Split** (chia ngẫu nhiên theo ảnh):
  - Khung hình ở giây thứ 1.0 sẽ nằm ở tập Train, còn khung hình ở giây thứ 1.5 nằm ở tập Test.
  - Mô hình chỉ cần "học vẹt" (memorize) bối cảnh nền mà không học được khả năng khái quát hóa (generalization) của hư hại mặt đường.
  - Kết quả F1 trên test sẽ **cao giả tạo (inflated F1 > 0.85)**, nhưng khi đem ra chạy trên tuyến đường mới thì F1 sụp đổ về dưới 0.20.

#### Giải pháp của đề bài:
Hàm `scripts/splitting.py` gom nhóm ảnh theo **phiên quay (session)** (dựa trên ngày giờ quay trong tên file, ví dụ `vlc-2025-02-18-23`, `phone-20250219`). Toàn bộ khung hình của cùng một phiên quay bắt buộc phải nằm trọn vẹn trong duy nhất một split (Train, Val hoặc Test).

---

### 8. Phân tích khoảng tin cậy Bootstrap CI95 trên tập Test 311 ảnh

#### Vấn đề thống kê:
Tập Test chỉ có 311 ảnh, bao gồm 121 pothole, 299 crack và 163 manhole.
- Khi kích thước mẫu nhỏ ($N = 311$), chỉ số $F_1$ điểm đơn (point estimate) chịu ảnh hưởng lớn từ phương sai mẫu ngẫu nhiên.
- Một mô hình có $F_1 = 0.52$ và một mô hình có $F_1 = 0.50$ hoàn toàn có thể không có sự khác biệt có ý nghĩa thống kê nếu khoảng tin cậy 95% của chúng chồng lấn (overlap) rộng rãi (ví dụ cả hai đều có $\text{CI95} \in [0.45, 0.57]$).

#### Khuyến nghị khoa học:
- Không bao giờ tuyên bố một setup tốt hơn setup khác nếu chỉ dựa vào chênh lệch $0.01$ hay $0.02$ điểm F1 khi CI95 chồng lấn nhau.
- Khoảng tin cậy rộng phản ánh sự biến thiên tự nhiên của tập dữ liệu thực tế (đặc biệt là lớp pothole có số lượng mẫu ít).

---

### 9. Cơ chế Verifiable Reward cho GRPO và phòng chống Reward Hacking

#### Hiện tượng Reward Hacking trong RL/GRPO:
Nếu hàm reward thiết kế không chặt chẽ, mô hình reinforcement learning sẽ tìm ra các "lối tắt" (shortcuts) để tối đa hóa reward mà không thực hiện đúng tác vụ:
1. **Shortcut "Dự đoán rỗng":** Nếu ảnh không có nhãn và model trả `[]` được 1.0 điểm, mô hình có thể suy thoái thành luôn luôn sinh `[]` cho mọi ảnh để tránh bị phạt lỗi sai.
2. **Shortcut "Box khổng lồ bao trùm":** Mô hình sinh một box cực lớn $[0, 0, 1000, 1000]$ để cố gắng nuốt trọn mọi box ground truth.
3. **Shortcut "Sinh hàng chục box bừa bãi":** Dự đoán dày đặc trên khung hình để vét trúng ground truth.

#### Thiết kế hàm Reward chuẩn mực (`src/reward.py`):
$$\text{Reward} = 0.2 \cdot \mathbb{I}(\text{JSON hợp lệ}) + 0.8 \cdot \frac{\sum_{(g, p) \in \text{Matches}} \text{IoU}(g, p)}{\max(N_{\text{gt}}, N_{\text{pred}}, 1)}$$
- Mẫu số $\max(N_{\text{gt}}, N_{\text{pred}}, 1)$ triệt tiêu ngay lập tức hành vi sinh thừa box (hallucination): nếu sinh 5 box thừa thì mẫu số tăng vọt khiến reward giảm mạnh.
- Ngưỡng $\text{IoU} \ge 0.5$ triệt tiêu hành vi sinh box khổng lồ bao trùm: box $[0, 0, 1000, 1000]$ có diện tích 1.000.000, trong khi ground truth trung bình chỉ chiếm diện tích ~7.000 $\rightarrow \text{IoU} \approx 0.007 < 0.5 \rightarrow \text{IoU} = 0$.
- Phạt format lỗi: JSON hỏng lập tức nhận `0.0`.

---

### 10. Chiến lược phòng chống mất phiên làm việc (Colab Disconnection Resilience)

Google Colab phiên bản miễn phí có thể ngắt kết nối (timeout / idle disconnect) sau 30–60 phút không tương tác hoặc khi sử dụng GPU liên tục.

#### 3 lớp bảo vệ tự động:
1. **Lưu định kỳ lên Google Drive:**
   Mỗi 100 steps huấn luyện, adapter và projector weights được tự động đồng bộ sang Google Drive tại thư mục `/content/drive/MyDrive/day10_checkpoints/`.
2. **Cơ chế Resume tất định trong suy luận:**
   Hàm `run_inference_on_split()` trong `src/inference.py` kiểm tra file `work/*.jsonl` hiện có. Nếu đã xử lý 150/311 ảnh, quá trình chạy lại sẽ tự động bỏ qua 150 ảnh đã có và chỉ chạy tiếp 161 ảnh còn lại, không lặp lại ảnh và không mất kết quả đã sinh.
3. **Kiểm tra trạng thái hoàn thành:**
   Mỗi experiment runner đọc `work/checkpoints/<setup>/run.json`. Nếu `global_step == 300`, pipeline tự động bỏ qua bước train để tránh tốn thời gian tính toán lại.
