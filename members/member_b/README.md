# Member B — ElasticNet & LightGBM

Chạy từ repo root, theo thứ tự:

```bash
python members/member_b/train_elasticnet.py   # ~40-60 phút (grid ~280 configs; có cache lại từng stage)
python members/member_b/train_lightgbm.py     # ~20-40 phút (tuning budget 40 configs)
python -m pytest tests/test_member_b.py -q
```

- `common.py`: nạp dữ liệu đúng contract (`read_dataset` → `holdout_indices`), scorer USD,
  wrapper giả thuyết `log1p` target, ghi metrics JSON + figures.
- Tuning/ablation **chỉ dùng 2344 dòng development** qua 5-fold KFold seed 123; mọi
  preprocessing fit lại trong từng training fold. **Test 586 dòng chỉ đánh giá đúng 1 lần**
  sau khi chốt cấu hình (khóa `n_test=586` được kiểm thử).
- ElasticNet: grid 2 stage (coarse → refine) + ablation identity-target + tham chiếu
  endpoint `l1_ratio=0/1` (Ridge/Lasso-like, chỉ để so sánh).
- LightGBM: RandomizedSearchCV 40 configs; importance theo gain; chẩn đoán tổng quát hóa
  theo neighborhood/yr_sold là hậu nghiệm, không ảnh hưởng lựa chọn.
- Kết quả: `results/member_b/{elasticnet,lightgbm}_metrics.json` (schema kiểm thử),
  bảng CV trong `results/member_b/*.csv`, hình trong `report/figures/{enet,lgbm}_*.png`.
- Không sửa dữ liệu chung; không dùng `data/fitted_previews/` cho CV.
