# Locally generated model files

The Streamlit modeling workflow writes `learnt_*.pkl` artifacts here after fitting a selected, CV-tuned configuration on the development partition.

The artifacts are ignored by Git (`*.pkl`). They can be recreated through the app's **Huấn luyện & lưu model** page or by the member training scripts. Each artifact includes fitted preprocessing and the estimator and must be loaded with compatible Python/library versions.
