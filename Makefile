# Convenience targets for scaffold smoke checks (no training / no downloads).

.PHONY: env data smoke dry-teacher dry-student help

help:
	@echo "make env          - scripts/00_check_env.py"
	@echo "make data         - scripts/01_verify_data_layout.py"
	@echo "make smoke        - scripts/02_smoke_import.py"
	@echo "make dry-teacher  - train_teacher --dry-run"
	@echo "make dry-student  - train_student_kd --dry-run"

env:
	python scripts/00_check_env.py

data:
	python scripts/01_verify_data_layout.py

smoke:
	python scripts/02_smoke_import.py

dry-teacher:
	python -m src.train.train_teacher --config configs/teacher_ny.yaml --dry-run

dry-student:
	python -m src.train.train_student_kd --config configs/student_kd_ny2dc.yaml --dry-run
