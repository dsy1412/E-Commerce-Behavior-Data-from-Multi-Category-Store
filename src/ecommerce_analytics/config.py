from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectConfig:
    project_root: Path = PROJECT_ROOT
    sample_modulus: int = 1_000
    sample_threshold: int = 20
    batch_size: int = 500_000
    observation_minutes: int = 5
    validation_fraction: float = 0.20
    random_state: int = 42
    run_shap: bool = True

    @property
    def raw_data_dir(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.project_root / "data" / "processed"

    @property
    def sampled_events_path(self) -> Path:
        return self.processed_data_dir / "sampled_events.parquet"

    @property
    def cleaned_events_path(self) -> Path:
        return self.processed_data_dir / "cleaned_events.parquet"

    @property
    def session_features_path(self) -> Path:
        return self.processed_data_dir / "session_features.parquet"

    @property
    def figures_dir(self) -> Path:
        return self.project_root / "reports" / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.project_root / "reports" / "tables"

    @property
    def outputs_dir(self) -> Path:
        return self.project_root / "outputs"

    @property
    def report_path(self) -> Path:
        return self.project_root / "reports" / "product_report.md"

    @property
    def experiment_path(self) -> Path:
        return self.project_root / "reports" / "experiment_plan.md"

    def ensure_directories(self) -> None:
        for path in (
            self.raw_data_dir,
            self.processed_data_dir,
            self.figures_dir,
            self.tables_dir,
            self.outputs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
