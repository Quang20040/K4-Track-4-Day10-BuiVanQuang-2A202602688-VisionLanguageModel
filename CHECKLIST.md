# BẢNG THEO DÕI TIÊU CHÍ ĐÁNH GIÁ (SUBMISSION CHECKLIST)
## VinUni AI Track 4 Day 10 — Finetune Qwen3-VL-4B bằng QLoRA trên Google Colab T4

**Học viên:** Bùi Văn Quang  
**Mã số sinh viên:** 2A202602688  
**Repository:** `https://github.com/Quang20040/K4-Track-4-Day10-BuiVanQuang-2A202602688-VisionLanguageModel`  

---

### PHẦN A. KẾT QUẢ CHẠY TỰ ĐỘNG (40 ĐIỂM)

| Tiêu chí | Điểm tối đa | Trạng thái kỹ thuật | Bằng chứng / File xác thực |
|---|---|---|---|
| **Đủ 4 file evidence `e0.json`–`e3.json` hợp lệ** | 12 | Sẵn sàng | `report/evidence/e0.json`, `e1.json`, `e2.json`, `e3.json` (`split="test"`, `missing_predictions=0`, `score_version=1`) |
| **Metadata đầy đủ cho E1–E3** | 6 | Sẵn sàng | Có đủ 3 trường số dương: `trainable_params`, `peak_vram_gb`, `train_minutes` |
| **Output hợp lệ của E1** | 4 | Sẵn sàng | `parse_valid_rate >= 0.95` (nhờ loss masking và canonical JSON formatting) |
| **E1 thật sự học được** | 6 | Sẵn sàng | $F_1(E1) - F_1(E0) \ge 0.10$ |
| **Bằng chứng E4 hợp lệ** | 6 | Sẵn sàng | Có đầy đủ `report/evidence/e4_data_0.1.json` và `report/evidence/e4_data_1.0.json` |
| **Bằng chứng nhất quán** | 6 | Sẵn sàng | Cùng `dataset_version` (`d10-21e4070301`) và khớp `data/splits/meta.json` |

---

### PHẦN B. THÍ NGHIỆM VÀ BẰNG CHỨNG (20 ĐIỂM)

| Tiêu chí | Điểm tối đa | Trạng thái kỹ thuật | Bằng chứng / File xác thực |
|---|---|---|---|
| **Bảng tổng hợp E0–E3** | 6 | Sẵn sàng | Đầy đủ $F_1$, CI95, FP/ảnh, VRAM đỉnh, thời gian huấn luyện, $F_1$ từng lớp (`pothole`, `crack`, `manhole`) |
| **Tối thiểu 6 ảnh minh hoạ có chú thích** | 4 | Sẵn sàng | `report/figures/prediction_01.png` đến `prediction_06.png` với đầy đủ chú thích True Positive, False Positive, False Negative, Multi-box, Subtle crack, Class confusion |
| **Thí nghiệm E4 sạch (Single-variable Ablation)** | 6 | Sẵn sàng | Đổi duy nhất biến `data_size` (10% vs 100%), cố định model, lr, steps, seed; phân tích dựa trên Bootstrap CI95 |
| **Nhật ký tài nguyên chính xác** | 4 | Sẵn sàng | VRAM và thời gian trong `REPORT.md` khớp 100% với metadata trong các file evidence |

---

### PHẦN C. BÁO CÁO (30 ĐIỂM)

| Câu hỏi | Điểm | Trọng tâm phân tích | Trạng thái |
|---|---|---|---|
| **Câu 1: Kiến trúc ViT, Projector, LLM** | 3 | Số tham số từng khối (~415M ViT, ~109M Projector, ~4.0B LLM); khối nào được cập nhật trong E1/E2/E3 | Đã hoàn thành |
| **Câu 2: E0 Zero-shot & 2 lỗi phổ biến** | 3 | Tỷ lệ parse hợp lệ của mô hình gốc, phân tích lỗi hallucination và sai lệch thang đo | Đã hoàn thành |
| **Câu 3: Thang đo 0–1000 vs Pixel (Chấm kỹ)** | 3 | Bản chất grounding tokens, tại sao sai thang đo dẫn đến F1 = 0, đối chiếu unit test | Đã hoàn thành |
| **Câu 4: Autoregressive Loss Masking (Chấm kỹ)** | 3 | Cơ chế `-100`, tại sao chỉ tính loss trên assistant tokens, nguy cơ khi tính trên prompt/image | Đã hoàn thành |
| **Câu 5: Huấn luyện Projector & Float32** | 3 | Trọng số 4-bit không có float gradient; tính ổn định số học của Float32 trong fp16 autocast | Đã hoàn thành |
| **Câu 6: Session-based Splitting (Chấm kỹ)** | 3 | Nguy cơ data leakage giữa các frame liền kề của dashcam; thống kê số ảnh/box từng split | Đã hoàn thành |
| **Câu 7: So sánh E1 / E2 / E3** | 3 | Trade-off giữa chi phí tài nguyên (VRAM, thời gian) và hiệu quả định vị thực tế | Đã hoàn thành |
| **Câu 8: Khoảng tin cậy Bootstrap CI95 (Chấm kỹ)** | 3 | Giới hạn thống kê trên 311 ảnh test, ý nghĩa của khoảng chồng lấn (overlapping CIs) | Đã hoàn thành |
| **Câu 9: Phân tích lỗi theo lớp** | 3 | Phân tích chi tiết tại sao `crack` khó định vị nhất, tác động của diện tích nhỏ đến IoU | Đã hoàn thành |
| **Câu 10: Phân tích Ablation E4** | 3 | So sánh 10% vs 100% dữ liệu train, tính hiệu quả dữ liệu (data efficiency) | Đã hoàn thành |

---

### PHẦN D. GIT VÀ TRÌNH BÀY (10 ĐIỂM)

| Tiêu chí | Điểm tối đa | Trạng thái kỹ thuật | Bằng chứng / File xác thực |
|---|---|---|---|
| **Git sạch và $\ge 5$ commits có ý nghĩa** | 5 | Sẵn sàng | Không commit `data/`, `work/`, `.env`, trọng số `*.safetensors`, `*.pt`; tối thiểu 5 commits tương ứng các phase |
| **`check_submission.py` 0 FAIL, 0 WARN** | 5 | Sẵn sàng | Toàn bộ kiểm tra tự động đạt chuẩn |

---

### PHẦN BONUS E5: GRPO (+10 ĐIỂM)

| Tiêu chí | Điểm tối đa | Trạng thái kỹ thuật | Bằng chứng / File xác thực |
|---|---|---|---|
| **Hàm Reward IoU có kiểm chứng & Unit tests** | 4 | ĐẠT 100% | `src/reward.py` và 7 test cases kiểm chứng biên trong `tests_student/test_reward.py` (Passed) |
| **Chạy GRPO training có log** | 3 | Sẵn sàng | Cấu hình `configs/e5.yaml` tích hợp cùng `trl.GRPOTrainer` |
| **Phân tích Reward Hacking & so sánh SFT** | 3 | Sẵn sàng | Phân tích cơ chế chống box bao trùm và shortcut rỗng qua mẫu số $\max(N_{\text{gt}}, N_{\text{pred}}, 1)$ |
