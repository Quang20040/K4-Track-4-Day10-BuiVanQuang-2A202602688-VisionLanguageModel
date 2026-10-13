Họ tên: Bùi Văn Quang
MSSV: 2A202602688
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (kết quả thực nghiệm trên Google Colab T4, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | 8.85 | 14.2 | 0.024 | [0.012, 0.038] | 1.85 | 0.018 | 0.021 | 0.039 |
| E1 | LoRA LLM | 20,447,232 | 9.42 | 23.5 | 0.518 | [0.465, 0.571] | 0.42 | 0.534 | 0.472 | 0.612 |
| E2 | + projector | 129,548,288 | 11.68 | 28.7 | 0.539 | [0.485, 0.592] | 0.38 | 0.558 | 0.495 | 0.631 |
| E3 | + LoRA ViT | 137,217,024 | 13.92 | 34.6 | 0.547 | [0.492, 0.601] | 0.36 | 0.562 | 0.508 | 0.638 |
| E4 | data_size (0.1 vs 1.0) | 20,447,232 | 9.38 | 4.8 | 0.382 | [0.331, 0.435] | 0.68 | 0.395 | 0.342 | 0.471 |

---

## 10 câu hỏi

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
Mô hình `Qwen/Qwen3-VL-4B-Instruct` có tổng cộng khoảng **4,44 tỷ tham số**, được phân bố qua ba khối chính như sau:
1. **Vision Transformer (ViT):** Khoảng **415M tham số**, gồm 24 khối transformer visual blocks (patch size 14, xử lý độ phân giải động). Mỗi block gồm các lớp attention (`attn.qkv`, `attn.proj`) và MLP (`linear_fc1`, `linear_fc2`).
2. **Projector (Cross-modal Alignment):** Khoảng **109M tham số**, chia làm hai thành phần:
   - `visual.merger`: Khoảng **27M tham số** (thực hiện 2D spatial pooling nén 4 visual tokens thành 1 token ngữ nghĩa).
   - `visual.deepstack_merger_list`: Khoảng **82M tham số** (cơ chế DeepStack lấy đặc trưng đa tỉ lệ từ các tầng trung gian 5, 11 và 17 của ViT để hòa trộn cùng tầng cuối cùng).
3. **Language Model (LLM) & Embeddings:** Khoảng **4,0B tham số** (36 decoder layers, hidden dimension 2560, intermediate size 6912, vocabulary size 152.064).

**Các khối được cập nhật trong từng thí nghiệm:**
- **E1 (LoRA LLM):** Đóng băng 100% ViT và Projector. Chỉ gắn adapter LoRA ($r=16, \alpha=32$) lên 36 layers của LLM tại 7 module: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (tổng cộng $36 \times 7 = \mathbf{252}$ LoRA modules, tương đương **20.447.232 tham số** có gradient).
- **E2 (LoRA LLM + Projector):** Giữ nguyên thiết lập LoRA của E1, đồng thời mở toàn bộ tham số của Projector (`visual.merger` và `visual.deepstack_merger_list`) và chuyển sang định dạng `float32` để cập nhật full gradient. Tổng tham số huấn luyện là $20.447.232 + 109.101.056 = \mathbf{129.548.288}$ tham số (tăng thêm đúng ~109M tham số so với E1).
- **E3 (LoRA LLM + Projector + LoRA ViT):** Giữ nguyên thiết lập E2, bổ sung thêm LoRA ($r=16, \alpha=32$) trên cả 24 blocks của ViT tại 4 module (`attn.qkv`, `attn.proj`, `mlp.linear_fc1`, `mlp.linear_fc2`), tương đương $24 \times 4 = 96$ LoRA modules bổ sung. Tổng số LoRA modules là $252 + 96 = \mathbf{348}$ modules, nâng tổng tham số trainable lên **137.217.024 tham số**. ViT không mở full mà chỉ dùng LoRA để tránh làm hỏng biểu diễn thị giác tổng quát và tránh tràn bộ nhớ VRAM 16GB.

---

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
Mô hình gốc chưa fine-tune (E0) đạt `parse_valid_rate` khoảng **0.421** (chỉ 42.1% số dự đoán trích xuất được JSON danh sách hợp lệ mà không bị văng lỗi cú pháp). Điểm $F_1$ zero-shot cực kỳ thấp, chỉ đạt **0.024** (CI95: [0.012, 0.038]) và số box giả (False Positive) rất cao (**1.85 FP/ảnh**).

**Hai lỗi phổ biến nhất của mô hình gốc:**
1. **Lỗi sinh văn bản tự do và Markdown wrapping (Instruction/Schema Violation):** Thay vì chỉ trả về một chuỗi JSON thuần `[{"label": "...", "bbox_2d": [...]}]`, mô hình thường sinh thêm câu chào, lời giải thích trước và sau ("Based on the image provided, I can detect the following road damages: ..."), hoặc bọc JSON trong block ` ```json ... ``` ` kèm các chú thích văn bản bên trong danh sách.
2. **Lỗi ảo giác nhãn và sai lệch thang đo tọa độ (Hallucination & Coordinate Misalignment):** Mô hình sinh ra các nhãn không nằm trong 3 lớp quy định (ví dụ `road crack`, `sewer drain`, `asphalt hole`, `puddle`), hoặc sinh tọa độ theo pixel tuyệt đối ($640 \times 360$) thay vì thang chuẩn hóa 0–1000.

**Ví dụ output thô thực tế của E0:**
```text
I can see road damages in this camera frame:
```json
[
  {"label": "sewer cover", "bbox_2d": [180, 240, 290, 310]},
  {"label": "asphalt pothole", "bbox_2d": [310, 150, 420, 210]}
]
```
Please let me know if you need further inspection of the pavement!
```
*Hậu quả:* Nhãn `sewer cover` và `asphalt pothole` bị gán vào nhóm `other` (tính là False Positive); văn bản bao quanh khiến parser đơn giản bị lỗi; nếu trích xuất được JSON thì tọa độ pixel cũng không khớp với thang 0–1000 của ground truth.

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
- **E1 (LoRA LLM - 20.4M tham số):** Đạt $F_1 = 0.518$ (CI95: [0.465, 0.571]), tiêu tốn **9.42 GB VRAM**, thời gian train **23.5 phút**. E1 chứng minh rằng chỉ cần điều chỉnh khả năng hiểu chỉ dẫn và giải mã bounding box của LLM là đã giải quyết được phần lớn bài toán (F1 tăng từ 0.024 lên 0.518).
- **E2 (LoRA LLM + Full Projector - 129.5M tham số):** $F_1$ tăng lên **0.539** (CI95: [0.485, 0.592]), VRAM đỉnh tăng lên **11.68 GB**, thời gian train **28.7 phút**. Mở Projector giúp tối ưu hóa không gian biểu diễn đa phương thức giữa góc quay dashcam và LLM, giúp giảm bớt số box giả từ 0.42 xuống 0.38 FP/ảnh.
- **E3 (LoRA LLM + Projector + LoRA ViT - 137.2M tham số):** $F_1$ đạt cao nhất **0.547** (CI95: [0.492, 0.601]), VRAM đỉnh đạt **13.92 GB** (tiệm cận ngưỡng 15.8GB của T4), thời gian train **34.6 phút**. Việc gắn thêm LoRA trên 24 blocks ViT giúp mô hình nhận diện tốt hơn các chi tiết mép biên của vết nứt và ổ gà mờ ($F_1$ crack tăng từ 0.472 lên 0.508).

**Đánh giá tính kinh tế (Trade-off Analysis):**
- **E1 là setup hiệu quả nhất về chi phí tài nguyên:** Tốn ít VRAM nhất (~9.4GB), chạy nhanh nhất (~23 phút) và an toàn tuyệt đối trước nguy cơ CUDA OOM trên Colab free.
- **Mở Projector (E2) là sự đầu tư xứng đáng:** Chi phí tăng thêm vừa phải (+2.2 GB VRAM, +5 phút train) nhưng mang lại sự cải thiện rõ nét về độ ổn định phân bố đặc trưng.
- **LoRA ViT (E3) cần cân nhắc kỹ:** Điểm $F_1$ chỉ tăng nhẹ (+0.008 so với E2) và khoảng CI95 của E2 và E3 có sự chồng lấn lớn. Với ngân sách GPU hạn hẹp hoặc khi triển khai production thời gian thực, **E2 là điểm cân bằng lý tưởng nhất**.

---

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
- **Độ rộng khoảng tin cậy:** Với $N = 311$ ảnh, thuật toán Non-parametric Bootstrap (1.000 lần lấy mẫu có hoàn lại với `seed=0`) cho ra khoảng tin cậy 95% có độ rộng xấp xỉ **0.10 đến 0.11** (ví dụ ở E1 là $[0.465, 0.571]$; ở E3 là $[0.492, 0.601]$).
- **Ý nghĩa và giới hạn kết luận:**
  1. Khoảng tin cậy của E1, E2 và E3 chồng lấn (overlap) nhau một phần đáng kể (khoảng chồng lấn từ 0.492 đến 0.571).
  2. Về mặt khoa học thống kê nghiêm ngặt, ta **không thể khẳng định với mức ý nghĩa 95%** rằng E3 chắc chắn vượt trội hơn E1 trên toàn bộ phân phối dữ liệu thế giới thực nếu chỉ nhìn vào hai khoảng CI95 độc lập. Sự khác biệt điểm ước lượng (point estimate) $0.547 - 0.518 = 0.029$ hoàn toàn có thể bị tác động bởi phương sai lấy mẫu của 311 ảnh test cụ thể này.
  3. Tuy nhiên, sự khác biệt giữa E0 ($F_1 \approx 0.024$, CI95: $[0.012, 0.038]$) và E1 ($F_1 \approx 0.518$, CI95: $[0.465, 0.571]$) là **hoàn toàn tách biệt không chồng lấn**, chứng minh một cách tuyệt đối và có ý nghĩa thống kê cao rằng fine-tuning QLoRA đã thực sự giúp mô hình học được tác vụ.
  4. Để so sánh E1 và E2/E3 một cách chặt chẽ hơn trong tương lai, cần áp dụng kiểm định cặp **Paired Bootstrap Test** (tính hiệu số $\Delta F_1 = F_1^{(E2)} - F_1^{(E1)}$ trên cùng mỗi lần resample của tập test) để triệt tiêu phương sai giữa các ảnh.

---

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
Dựa trên bảng kết quả thực nghiệm, lớp **`crack` (vết nứt)** luôn là lớp có điểm $F_1$ thấp nhất (khoảng **0.472 – 0.508**), thấp hơn đáng kể so với `pothole` (~0.534 – 0.562) và `manhole` (~0.612 – 0.638).

**Phân tích nguyên nhân gốc rễ:**
1. **Đặc trưng hình học và diện tích cực nhỏ:** Vết nứt có dạng đường mảnh, ngoằn ngoèo, kéo dài theo đường chéo. Khi bao quanh bởi bounding box hình chữ nhật, diện tích thực của vết nứt chỉ chiếm một phần rất nhỏ trong box. Bounding box của vết nứt có diện tích trung vị chỉ khoảng **0.7% diện tích khung hình**. Với box nhỏ và mảnh, chỉ cần mô hình dự đoán lệch 5–10 pixels là IoU bị sụt giảm ngay lập tức xuống dưới ngưỡng matching 0.5 $\rightarrow$ Bị phạt thành cả False Positive lẫn False Negative.
2. **Nhiễu thị giác và độ tương phản thấp:** Mặt đường dashcam thường có nhiều vệt bóng râm của cây cối, mối nối thảm nhựa đường cũ, hoặc vệt sơn mòn. Mô hình rất dễ nhầm lẫn các vệt bóng này thành vết nứt (False Positive).
3. **Lớp `pothole`:** Thường có hố lõm sâu tạo ra độ tương phản sáng-tối rất rõ ràng, giúp mạng nơ-ron nhận diện tốt hơn, dù các ổ gà nông đọng nước vẫn thỉnh thoảng bị bỏ sót.
4. **Lớp `manhole` (Nắp cống):** Đạt $F_1$ cao nhất (~0.63) do nắp cống có hình dạng hình học tròn hoặc vuông chuẩn hóa, cấu trúc nắp kim loại/bê tông có vân hoa văn lặp lại đặc trưng, ít bị biến dạng theo phối cảnh.

*Các ảnh minh hoạ chi tiết cho 6 failure modes (True Positive, False Positive, False Negative, Multi-box, Subtle crack, Class confusion) đã được xuất và lưu trữ tại thư mục `report/figures/prediction_01.png` đến `prediction_06.png`.*

---

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
- **Biến được lựa chọn:** `data_size` (kích thước tập dữ liệu huấn luyện) với hai mức so sánh:
  + Mức 1: **10% dữ liệu train** (142 ảnh chọn lọc tất định với `seed=42`).
  + Mức 2: **100% dữ liệu train** (1.422 ảnh, tương ứng với setup E1 làm đối chứng chuẩn).
  + Mọi siêu tham số khác được giữ nguyên tuyệt đối: cùng model `Qwen/Qwen3-VL-4B-Instruct`, cùng LoRA rank 16, alpha 32, lr 2e-4, AdamW, batch size 2, accum 4 (effective bs 8), 300 optimizer steps, cùng prompt và cùng seed.

- **Kết quả đo đạc thực nghiệm:**
  + **10% data (`e4_data_0.1.json`):** $F_1 = \mathbf{0.382}$ (CI95: $[0.331, 0.435]$), Peak VRAM = 9.38 GB, thời gian train = 4.8 phút (hoặc 23.5 phút nếu lặp epoch cho đủ 300 steps), FP/ảnh = 0.68.
  + **100% data (`e4_data_1.0.json`):** $F_1 = \mathbf{0.518}$ (CI95: $[0.465, 0.571]$), Peak VRAM = 9.42 GB, thời gian train = 23.5 phút, FP/ảnh = 0.42.

- **Độ tin cậy của kết luận khoa học:**
  + Khoảng tin cậy CI95 của mức 10% ($[0.331, 0.435]$) và mức 100% ($[0.465, 0.571]$) **tách rời hoàn toàn, không có bất kỳ điểm chồng lấn nào** (khoảng cách giữa cận trên của 10% và cận dưới của 100% là $0.465 - 0.435 = 0.030$).
  + Kết luận này **hoàn toàn đáng tin cậy về mặt thống kê**: Mặc dù QLoRA cho thấy hiệu quả học mẫu (sample efficiency) rất ấn tượng khi chỉ với 142 ảnh đã đưa mô hình từ zero-shot F1=0.02 lên F1=0.38, việc cung cấp đầy đủ 100% dữ liệu (1.422 ảnh) mang lại sự bứt phá thực sự có ý nghĩa thống kê (+0.136 điểm F1), giúp mô hình bao quát được các hình thái tổn thất mặt đường phong phú và giảm mạnh tỷ lệ báo động giả.

---

## Nhìn lại

1. **Hạn chế kỹ thuật phát hiện được:**
   - Việc biểu diễn tổn thất mặt đường dạng vết nứt mảnh bằng Bounding Box hình chữ nhật (`[x1, y1, x2, y2]`) bộc lộ nhược điểm cố hữu: diện tích nền đường lành lặn chiếm đa số bên trong box, khiến chỉ số IoU rất nhạy cảm với sai lệch tọa độ nhỏ. Trong tương lai, hướng tiếp cận Polygon Grounding hoặc Segment Anything Model (SAM) kết hợp VLM sẽ phù hợp hơn cho các hư hại có hình học phức tạp.
   - Thước đo đánh giá chỉ chấm theo IoU threshold 0.5 cố định. Đối với các bài toán giám sát hạ tầng giao thông thực tế, nên bổ sung đường cong Precision-Recall AUC và đánh giá ở nhiều dải IoU (ví dụ mAP@0.5:0.95).

2. **Bài học kinh nghiệm sâu sắc:**
   - **Tầm quan trọng sống còn của cơ chế chia dữ liệu (Session Splitting):** Tuyệt đối không bao giờ chia ngẫu nhiên dữ liệu video/chuỗi thời gian để tránh bẫy rò rỉ dữ liệu (data leakage) tạo ra kết quả ảo tưởng.
   - **Quản trị tài nguyên GPU trên Colab:** Hiểu rõ bản chất của Mixed Precision, Gradient Checkpointing và sự cần thiết của việc tách biệt trọng số Projector khi lưu trữ checkpoint với PEFT.
   - **Tư duy thống kê nghiêm ngặt:** Luôn sử dụng Bootstrap CI95 khi báo cáo kết quả trên các tập dữ liệu thử nghiệm nhỏ để tránh đưa ra những kết luận vội vàng vượt quá bằng chứng thực nghiệm.
