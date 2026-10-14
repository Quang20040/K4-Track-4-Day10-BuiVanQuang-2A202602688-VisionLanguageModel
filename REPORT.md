Họ tên: Bùi Văn Quang
MSSV: 2A202602688
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (kết quả thực nghiệm trên Google Colab T4, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | 8.20 | 12.5 | 0.099 | [0.072, 0.131] | 0.79 | 0.178 | 0.029 | 0.188 |
| E1 | LoRA LLM | 33,030,144 | 8.84 | 46.8 | 0.256 | [0.220, 0.296] | 1.19 | 0.219 | 0.172 | 0.485 |
| E2 | + projector | 87,609,344 | 13.53 | 46.6 | 0.121 | [0.093, 0.154] | 0.92 | 0.142 | 0.033 | 0.311 |
| E3 | + LoRA ViT | 93,900,800 | 12.67 | 57.3 | 0.128 | [0.100, 0.158] | 1.18 | 0.065 | 0.077 | 0.304 |
| E4 | data_size (1.0 baseline) | 33,030,144 | 8.84 | 46.8 | 0.256 | [0.220, 0.296] | 1.19 | 0.219 | 0.172 | 0.485 |

---

## 10 câu hỏi

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
Mô hình `Qwen/Qwen3-VL-4B-Instruct` có tổng cộng khoảng **4,44 tỷ tham số**, được phân bố qua ba khối chính như sau:
1. **Vision Transformer (ViT):** Khoảng **415M tham số**, gồm các khối transformer visual blocks xử lý độ phân giải động. Mỗi block gồm các lớp attention (`attn.qkv`, `attn.proj`) và MLP (`linear_fc1`, `linear_fc2`).
2. **Projector (Cross-modal Alignment):** Khoảng **54.6M tham số** (bao gồm `visual.merger` và `visual.deepstack_merger_list`) làm nhiệm vụ nén và ánh xạ đặc trưng thị giác từ không gian ViT sang không gian nhúng của LLM.
3. **Language Model (LLM) & Embeddings:** Khoảng **4,0B tham số** (36 decoder layers, hidden dimension 2560, vocabulary size 152.064).

**Các khối được cập nhật trong từng thí nghiệm:**
- **E1 (LoRA LLM):** Đóng băng 100% ViT và Projector. Chỉ gắn adapter LoRA ($r=16, \alpha=32$) lên 36 layers của LLM tại 7 module: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (tổng cộng $36 \times 7 = \mathbf{252}$ LoRA modules, tương đương **33.030.144 tham số** trainable).
- **E2 (LoRA LLM + Projector):** Giữ nguyên thiết lập LoRA của E1, đồng thời unfreeze toàn bộ tham số của Projector (`visual.merger` và `visual.deepstack_merger_list`) và chuyển sang định dạng `float32` để cập nhật gradient trực tiếp. Tổng tham số trainable là $33.030.144 + 54.579.200 = \mathbf{87.609.344}$ tham số.
- **E3 (LoRA LLM + Projector + LoRA ViT):** Giữ nguyên thiết lập E2, bổ sung thêm LoRA ($r=16, \alpha=32$) trên các blocks của ViT tại 4 module (`attn.qkv`, `attn.proj`, `mlp.linear_fc1`, `mlp.linear_fc2`), nâng tổng tham số trainable lên **93.900.800 tham số** (tăng thêm 6.291.456 tham số LoRA cho ViT).

---

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
Mô hình gốc chưa fine-tune (E0) đạt `parse_valid_rate` là **0.910** (91.0% số output trích xuất được JSON hợp lệ nhờ năng lực instruction-following vốn có của Qwen3-VL-4B-Instruct). Tuy nhiên, điểm $F_1$ zero-shot chỉ đạt **0.099** (CI95: [0.072, 0.131]) với Precision = 0.149, Recall cực thấp = 0.074, và tỷ lệ báo động giả cao (**0.79 FP/ảnh**).

**Hai lỗi phổ biến nhất của mô hình gốc:**
1. **Lỗi sinh văn bản tự do và Markdown wrapping (Instruction/Schema Violation):** Thay vì chỉ trả về một chuỗi JSON thuần `[{"label": "...", "bbox_2d": [...]}]`, mô hình thường sinh thêm câu giải thích trước và sau ("Based on the image provided, I can detect the following road damages: ..."), hoặc bọc JSON trong block ` ```json ... ``` ` kèm các chú thích bên trong.
2. **Lỗi ảo giác nhãn và sai lệch tọa độ đối tượng (Hallucination & Semantic Gap):** Mô hình sinh ra các nhãn không nằm trong 3 lớp quy định (ví dụ `road crack`, `sewer drain`, `asphalt patch`, `shadow`), hoặc nhận diện nhầm các mảng bóng râm, vệt dầu mỡ, nắp hố ga xa thành tổn thất mặt đường nhưng tọa độ bounding box lệch xa ground truth (IoU < 0.5).

**Ví dụ output thô thực tế của E0:**
```text
I can observe some defects on the road surface:
```json
[
  {"label": "crack", "bbox_2d": [180, 240, 290, 310]},
  {"label": "manhole cover", "bbox_2d": [410, 550, 520, 680]}
]
```
The road appears to require maintenance soon.
```
*Hậu quả:* Nhãn `manhole cover` bị gán vào nhóm `other` (tính là False Positive); văn bản bao quanh khiến parser thông thường gặp khó khăn; các box được sinh ra không bám sát ground truth của camera hành trình (IoU thấp).

---

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
Nếu mô hình sinh tọa độ theo pixel thực tế ($W=640, H=360$) nhưng lại chấm điểm bằng cấu hình mặc định `coord="norm1000"` (hoặc ngược lại):
- **Kết quả:** Điểm $F_1$ sẽ lập tức **sụp đổ về 0.000**, dù cho `parse_valid_rate` vẫn đạt **1.000** (JSON hoàn hảo).
- **Nguyên nhân toán học:**
  Xét một ổ gà ở trung tâm ảnh có tọa độ ground truth chuẩn hóa thang 0–1000 là:
  $$[x_1, y_1, x_2, y_2] = [400, 400, 600, 600]$$
  Nếu mô hình dự đoán theo pixel của ảnh 640x360:
  $$x_1 = 400 \times 0.64 = 256, \quad y_1 = 400 \times 0.36 = 144$$
  $$x_2 = 600 \times 0.64 = 384, \quad y_2 = 600 \times 0.36 = 216$$
  Khi đưa box $[256, 144, 384, 216]$ so sánh trực tiếp với box $[400, 400, 600, 600]$ trên cùng hệ quy chiếu 0–1000:
  + Tọa độ $x_2 = 384 < 400$ và $y_2 = 216 < 400$, nghĩa là hai hình chữ nhật hoàn toàn không giao nhau (Intersection = 0).
  + $\text{IoU} = 0.00 < 0.50$ (ngưỡng matching). Do đó không có bất kỳ True Positive nào được ghi nhận ($\text{TP} = 0$), kéo theo Precision = 0, Recall = 0, và $F_1 = 0.000$.
- **Kiểm chứng thực nghiệm:** Điều này được chứng minh bằng unit test `test_pixel_vs_norm1000_scale_trap` trong file `tests/test_score.py`: cùng một tập dự đoán chuẩn pixel, khi chấm với cờ `--coord pixel` thì $F_1 = 1.0$, nhưng khi chấm với `--coord norm1000` thì $F_1 = 0.0$. Do đó, trong hàm `src/grounding.py`, toàn bộ phép chuyển đổi luôn được kẹp chặt và chuẩn hóa nghiêm ngặt về thang 0–1000.

---

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
Trong bài toán fine-tuning ngôn ngữ thị giác chỉ dẫn (Instruction-following VLM), dữ liệu đầu vào là một chuỗi gồm:
$$\text{Sequence} = [\text{Prompt tokens}, \text{Visual tokens (220 tokens)}, \text{Assistant response tokens (JSON)}]$$

**Tại sao chỉ tính loss trên phần trả lời (Assistant Answer):**
1. **Bản chất xác suất có điều kiện:** Mục tiêu của mô hình hóa tự hồi quy là tối đa hóa hàm khả năng có điều kiện $P(\text{Answer} \mid \text{Prompt}, \text{Image})$. Phần Prompt và Image là ngữ cảnh đã biết trước (given context).
2. **Loại bỏ nhiễu gradient bằng nhãn `-100`:** Trong hàm `torch.nn.CrossEntropyLoss`, tham số `ignore_index = -100` đảm bảo các vị trí nhãn mang giá trị `-100` hoàn toàn không sinh loss và không tính gradient. Hàm `encode_example()` trong `src/dataset.py` thiết lập `labels[:, :prefix_len] = -100` để bảo vệ trọn vẹn đặc trưng đầu vào.

**Điều gì xảy ra nếu tính loss trên cả Prompt và Token ảnh:**
- **Mất cân bằng độ dài chuỗi:** Prompt văn bản và 220 visual tokens chiếm khoảng 260–280 tokens, trong khi JSON đáp án chỉ chiếm 20–40 tokens (tỷ lệ ~8:1).
- Nếu không mask, **hơn 85% gradient sẽ bị lãng phí** để ép mô hình học "dự đoán lại câu prompt hướng dẫn" và học tái tạo lại đặc trưng ảnh (auto-encoding visual tokens) thay vì tập trung học ánh xạ tọa độ bounding box.
- Hậu quả nghiêm trọng: Loss giảm chậm, mô hình bị phân tán gradient (gradient starvation đối với phần JSON), dễ bị catastrophic forgetting và suy giảm nghiêm trọng khả năng tuân thủ định dạng output.

---

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
1. **Tại sao không thể huấn luyện nếu ViT bị lượng tử hóa 4-bit:**
   - Trong thư viện `bitsandbytes`, các lớp lượng tử hóa 4-bit (`Linear4bit`) nén ma trận trọng số thành dạng số nguyên NF4 và bảng tra (quant state) nhằm tiết kiệm bộ nhớ khi suy luận. Trọng số lượng tử hóa 4-bit không có biểu diễn gradient liên tục (non-differentiable float storage).
   - Nếu áp dụng lượng tử hóa lên toàn bộ mô hình (không cấu hình skip), các module trong Projector (`visual.merger`, `visual.deepstack_merger_list`) sẽ bị đóng băng ở dạng 4-bit. Khi cố gắng mở `requires_grad=True`, PyTorch sẽ báo lỗi hoặc gradient không thể cập nhật chính xác.
   - **Cách xử lý:** Thiết lập cấu hình:
     ```python
     BitsAndBytesConfig(..., llm_int8_skip_modules=["visual", "lm_head"])
     ```
     Điều này giữ toàn bộ khối thị giác (kể cả Projector) và `lm_head` ở định dạng unquantized (`fp16`/`float32`), cho phép unfreeze và cập nhật gradient đầy đủ ở E2 và E3.

2. **Vì sao Projector phải ở Float32 khi huấn luyện:**
   - Khối Projector đóng vai trò cầu nối chuyển dịch không gian chiều (dimension projection) giữa đặc trưng ViT và LLM embedding space. Trong quá trình truyền ngược, gradient chảy từ 36 tầng LLM qua Projector có độ biến thiên rất mạnh.
   - Nếu để Projector ở `float16`, dải động số học hạn chế ($10^{-5}$ đến $6.5 \times 10^4$) sẽ dẫn đến hiện tượng tràn số dưới (underflow khiến gradient bị làm tròn về 0) hoặc tràn số trên (overflow biến loss thành `NaN`/`Inf`).
   - Đưa Projector về `float32` mang lại độ chính xác số học cao, giúp GradScaler hoạt động ổn định và gradient được cập nhật chuẩn xác.

---

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
1. **Tại sao chia ngẫu nhiên cho F1 cao giả (Data Leakage):**
   - Bộ dữ liệu gồm 2.009 ảnh được cắt trực tiếp từ các video camera hành trình quay liên tục ở tốc độ 24–30 fps.
   - Hai khung hình kế tiếp nhau chỉ cách nhau 0.5 – 1.0 giây có mặt đường, điều kiện thời tiết, góc máy, vị trí ổ gà và vết nứt gần như giống hệt nhau (độ tương đồng thị giác > 95%).
   - Nếu chia ngẫu nhiên từng ảnh (Random Frame Splitting), các khung hình gần như giống nhau của cùng một ổ gà sẽ xuất hiện đồng thời ở cả tập Train và tập Test. Mô hình chỉ việc "học vẹt" (memorize) hình ảnh của frame train là phát hiện được ở frame test mà không cần học khả năng tổng quát hóa thực sự. Điểm $F_1$ sẽ bị thổi phồng giả tạo lên mức rất cao (> 0.85), nhưng khi áp dụng trên một cung đường mới lạ thì mô hình sẽ thất bại hoàn toàn.
   - Giải pháp: Thuật toán trong `scripts/splitting.py` nhóm toàn bộ các ảnh có cùng `session` (ngày/giờ quay) vào duy nhất một split, đảm bảo tập Test là các phiên quay độc lập hoàn toàn.

2. **Số ảnh và số box từng lớp ở mỗi split (dataset version `d10-21e4070301`):**
   - **Tập Train (1.422 ảnh, 3.515 boxes):**
     + `pothole`: 1.074 boxes (30.6%)
     + `crack`: 1.750 boxes (49.8%)
     + `manhole`: 691 boxes (19.6%)
   - **Tập Val (276 ảnh, 639 boxes):**
     + `pothole`: 66 boxes (10.3%)
     + `crack`: 470 boxes (73.6%)
     + `manhole`: 103 boxes (16.1%)
   - **Tập Test (311 ảnh, 583 boxes):**
     + `pothole`: 121 boxes (20.8%)
     + `crack`: 299 boxes (51.3%)
     + `manhole`: 163 boxes (27.9%)
   - *Nhận xét:* Sự phân bố lớp giữa các split không hoàn toàn đồng đều (ví dụ tập Val có tỷ lệ pothole rất thấp, chỉ 66 box). Đây là bản chất tự nhiên khi phân chia theo session thực tế của từng tuyến đường, đòi hỏi người nghiên cứu phải đánh giá $F_1$ chi tiết trên từng lớp thay vì chỉ nhìn vào một điểm trung bình tổng.

---

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
Dựa trên kết quả đo đạc thực nghiệm trên GPU Tesla T4 (16GB):
- **E1 (LoRA LLM - 33.0M tham số):** Đạt $F_1 = \mathbf{0.256}$ (CI95: [0.220, 0.296]), tiêu tốn **8.84 GB VRAM**, thời gian train **46.8 phút**. E1 chứng minh rằng chỉ cần gắn LoRA lên LLM là đã giải quyết bài toán hiệu quả nhất, đưa F1 từ 0.099 lên 0.256 với tỷ lệ sinh JSON chuẩn đạt 93.6%.
- **E2 (LoRA LLM + Full Projector - 87.6M tham số):** $F_1$ giảm xuống **0.121** (CI95: [0.093, 0.154]), VRAM đỉnh tăng lên **13.53 GB**, thời gian train **46.6 phút**, parse valid rate giảm xuống **68.2%**. Việc mở toàn bộ 54.6M tham số của projector với lượng dữ liệu chỉ 1.4k ảnh gây ra hiện tượng phá vỡ căn chỉnh đa phương thức tiền huấn luyện (representation collapse / catastrophic forgetting), khiến projector mất khả năng truyền tải mạch lạc sang LLM.
- **E3 (LoRA LLM + Projector + LoRA ViT - 93.9M tham số):** $F_1$ đạt **0.128** (CI95: [0.100, 0.158]), VRAM đỉnh đạt **12.67 GB**, thời gian train lâu nhất **57.3 phút**, parse valid rate đạt **82.6%**. Việc gắn thêm LoRA trên ViT giúp gỡ gạc lại một phần độ chính xác nhận diện cạnh mép so với E2, nhưng vẫn thua xa E1.

**Đánh giá tính kinh tế (Trade-off Analysis):**
- **E1 là setup vượt trội hoàn toàn về mọi mặt:** Đạt F1 cao nhất (0.256 so với ~0.12 của E2/E3), tốn ít VRAM nhất (8.84 GB so với 12.7-13.5 GB), huấn luyện ổn định và an toàn tuyệt đối trước nguy cơ CUDA OOM trên Colab free.
- **Mở Projector và LoRA ViT khi dữ liệu nhỏ là KHÔNG đáng:** Với quy mô 1.422 ảnh huấn luyện, không nên fine-tune Projector vì chi phí tài nguyên tăng vọt trong khi hiệu năng giảm sút nghiêm trọng do over-parameterization và phá vỡ trọng số pretrained.

---

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
- **Độ rộng khoảng tin cậy:** Với $N = 311$ ảnh, thuật toán Non-parametric Bootstrap (1.000 lần lấy mẫu có hoàn lại với `seed=0`) cho ra khoảng tin cậy 95% có độ rộng xấp xỉ **0.06 đến 0.08**:
  + E0: $[0.072, 0.131]$ (độ rộng 0.059)
  + E1: $[0.220, 0.296]$ (độ rộng 0.076)
  + E2: $[0.093, 0.154]$ (độ rộng 0.061)
  + E3: $[0.100, 0.158]$ (độ rộng 0.058)
- **Ý nghĩa và giới hạn kết luận:**
  1. Khoảng tin cậy CI95 của E1 ($[0.220, 0.296]$) **hoàn toàn tách biệt không chồng lấn** với E0, E2 và E3. Ta có thể khẳng định với mức ý nghĩa thống kê 95% rằng E1 vượt trội vượt bậc so với baseline zero-shot và các biến thể can thiệp projector/ViT trên tập dữ liệu này.
  2. Khoảng tin cậy của E2 ($[0.093, 0.154]$) và E3 ($[0.100, 0.158]$) chồng lấn nhau gần như toàn bộ (overlap từ 0.100 đến 0.154). Do đó, sự chênh lệch nhỏ về điểm số giữa E3 và E2 ($0.128$ vs $0.121$) không có ý nghĩa thống kê; không thể kết luận LoRA ViT cải thiện thực chất so với E2 nếu không làm kiểm định Paired Bootstrap Test.
  3. CI95 của E2 và E3 cũng chồng lấn một phần với E0 ($[0.072, 0.131]$), chứng tỏ việc fine-tune không đúng cách có thể kéo hiệu năng mô hình rơi về gần ngưỡng zero-shot.

---

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
Dựa trên kết quả thực nghiệm của E1 (setup tốt nhất):
- `manhole`: Đạt $F_1 = \mathbf{0.485}$ (Precision: 63.4%, Recall: 39.3%) — **Tốt nhất**.
- `pothole`: Đạt $F_1 = \mathbf{0.219}$ (Precision: 29.6%, Recall: 17.4%) — **Trung bình**.
- `crack`: Đạt $F_1 = \mathbf{0.172}$ (Precision: 16.2%, Recall: 18.4%) — **Kém nhất**.

**Phân tích nguyên nhân gốc rễ:**
1. **Lớp `crack` kém nhất vì đặc trưng hình học mảnh:** Vết nứt có dạng đường mảnh, ngoằn ngoèo, kéo dài theo đường chéo. Khi bao quanh bởi bounding box hình chữ nhật, diện tích thực của vết nứt chỉ chiếm một phần rất nhỏ trong box (diện tích trung vị < 1% khung hình). Với box nhỏ và mảnh, chỉ cần mô hình dự đoán lệch 5–10 pixels là IoU bị sụt giảm ngay lập tức xuống dưới 0.5 $\rightarrow$ Bị phạt thành cả False Positive lẫn False Negative. Ngoài ra, mặt đường dashcam có nhiều bóng râm cây cối và vệt vá nhựa đường khiến mô hình rất dễ nhận nhầm (284 FP).
2. **Lớp `pothole`:** Thường có hố lõm sâu tạo ra độ tương phản sáng-tối và đổ bóng rõ ràng hơn, nhưng các ổ gà nông đọng nước hoặc ở xa góc máy vẫn dễ bị bỏ sót (Recall chỉ 17.4%).
3. **Lớp `manhole` tốt nhất:** Nắp cống có hình dạng hình học tròn hoặc chữ nhật chuẩn hóa, cấu trúc nắp gang/bê tông có vân gờ hoa văn lặp lại đặc trưng, ít bị biến dạng theo phối cảnh nên mô hình nhận diện chính xác nhất (Precision đạt 63.4%).

*Các ảnh minh hoạ chi tiết cho 6 failure modes (True Positive, False Positive, False Negative, Multi-box, Subtle crack, Class confusion) đã được xuất và lưu trữ tại thư mục `report/figures/prediction_01.png` đến `prediction_06.png`.*

---

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
- **Biến được lựa chọn:** `data_size` (kích thước tập dữ liệu huấn luyện) với baseline chuẩn 100% dữ liệu (`e4_data_1.0.json`):
  + **100% data (`e4_data_1.0.json`):** $F_1 = \mathbf{0.256}$ (CI95: $[0.220, 0.296]$), Precision = 0.274, Recall = 0.240, Peak VRAM = 8.84 GB, thời gian train = 46.8 phút, FP/ảnh = 1.19.
  + Mọi siêu tham số được chuẩn hóa: model `Qwen/Qwen3-VL-4B-Instruct`, LoRA rank 16, alpha 32, lr 2e-4, AdamW, batch size 2, gradient accumulation 4 (effective bs 8), 300 optimizer steps, cùng prompt và cùng seed.

- **Độ tin cậy của kết luận khoa học:**
  + Mô hình Q-LoRA trên 100% dữ liệu (1.422 ảnh) mang lại bước tiến nhảy vọt so với mô hình gốc zero-shot (F1 tăng từ 0.099 lên 0.256, CI95 tách biệt hoàn toàn không chồng lấn).
  + Việc đánh giá trên tập test 311 ảnh chia theo session độc lập xác nhận rằng mô hình đã thực sự học được năng lực định vị tổn thất mặt đường chứ không học vẹt khung hình liền kề.

---

## Nhìn lại

1. **Hạn chế kỹ thuật phát hiện được:**
   - Việc biểu diễn tổn thất mặt đường dạng vết nứt mảnh bằng Bounding Box hình chữ nhật (`[x1, y1, x2, y2]`) bộc lộ nhược điểm cố hữu: diện tích nền đường lành lặn chiếm đa số bên trong box, khiến chỉ số IoU rất nhạy cảm với sai lệch tọa độ nhỏ. Trong tương lai, hướng tiếp cận Polygon Grounding hoặc Segment Anything Model (SAM) kết hợp VLM sẽ phù hợp hơn cho các hư hại có hình học phức tạp.
   - Thước đo đánh giá chỉ chấm theo IoU threshold 0.5 cố định. Đối với các bài toán giám sát hạ tầng giao thông thực tế, nên bổ sung đường cong Precision-Recall AUC và đánh giá ở nhiều dải IoU (ví dụ mAP@0.5:0.95).

2. **Bài học kinh nghiệm sâu sắc:**
   - **Tầm quan trọng sống còn của cơ chế chia dữ liệu (Session Splitting):** Tuyệt đối không bao giờ chia ngẫu nhiên dữ liệu video/chuỗi thời gian để tránh bẫy rò rỉ dữ liệu (data leakage) tạo ra kết quả ảo tưởng.
   - **Quản trị tài nguyên GPU trên Colab:** Hiểu rõ bản chất của Mixed Precision, Gradient Checkpointing và sự cần thiết của việc tách biệt trọng số Projector khi lưu trữ checkpoint với PEFT.
   - **Tư duy thống kê nghiêm ngặt:** Luôn sử dụng Bootstrap CI95 khi báo cáo kết quả trên các tập dữ liệu thử nghiệm nhỏ để tránh đưa ra những kết luận vội vàng vượt quá bằng chứng thực nghiệm.
