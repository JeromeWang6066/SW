# Convenience targets for scaffold smoke checks (no training / no downloads).

.PHONY: env data smoke shapes test dry-teacher dry-student help

help:
	@echo "make env          - scripts/00_check_env.py"
	@echo "make data         - scripts/01_verify_data_layout.py"
	@echo "make smoke        - scripts/02_smoke_import.py"
	@echo "make shapes       - scripts/03_synthetic_shapes.py"
	@echo "make test         - pytest tests/ (shape contracts)"
	@echo "make dry-teacher  - train_teacher --dry-run"
	@echo "make dry-student  - train_student_kd --dry-run"

env:
	python3 scripts/00_check_env.py

data:
	python3 scripts/01_verify_data_layout.py

smoke:
	python3 scripts/02_smoke_import.py

shapes:
	python3 scripts/03_synthetic_shapes.py

test:
	python3 -m pytest tests/ -q

dry-teacher:
	python3 -m src.train.train_teacher --config configs/teacher_ny.yaml --dry-run

dry-student:
	python3 -m src.train.train_student_kd --config configs/student_kd_ny2dc.yaml --dry-run
