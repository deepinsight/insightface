"""Reports page."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem

from ..core.exporters import export_csv
from ..core.licensing import model_license_display
from ..core.models import EvaluationResult
from ..core.reporting import write_reports
from ..core.utils import timestamp_for_filename, utc_now_iso
from ..widgets.table_utils import configure_table_columns, refresh_table_columns
from .base import BasePage


class ReportsPage(BasePage):
    def __init__(self, context, parent=None):
        super().__init__(context, "Reports", "Browse and export historical enterprise evaluation runs.", parent)
        self.runs = []
        self.content.addWidget(self.row(self.button("Refresh", self.refresh), self.button("Open Selected Report", self.open_selected), self.button("Re-export Selected", self.reexport_selected), self.button("Delete Selected", self.delete_selected), self.button("Export All Summary", self.export_summary)))
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["id", "date", "scenario", "model", "provider", "threshold", "report_path"])
        configure_table_columns(self.table, [60, 180, 180, 130, 110, 90, 360])
        self.content.addWidget(self.table, 1)
        self.refresh()

    def refresh(self) -> None:
        self.runs = self.context.storage.list_evaluation_runs()
        self.table.setRowCount(len(self.runs))
        for row_index, run in enumerate(self.runs):
            values = [run["id"], run["created_at"], run["scenario"], run["model_name"], run["provider"], run["threshold"], run["report_path"]]
            for col, value in enumerate(values):
                self.table.setItem(row_index, col, QTableWidgetItem(str(value)))
        refresh_table_columns(self.table)

    def _selected_run(self):
        row = self.table.currentRow()
        if row < 0 or row >= len(self.runs):
            self.show_error("Select a report row first.")
            return None
        return self.runs[row]

    def open_selected(self) -> None:
        run = self._selected_run()
        if not run:
            return
        path = run.get("report_path") or ""
        if not path:
            self.show_error("This run has no report path.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def reexport_selected(self) -> None:
        run = self._selected_run()
        if not run:
            return
        license_status = model_license_display(
            self.context.config,
            model_name=str(run["model_name"]),
        ).status_text
        result = EvaluationResult(scenario=run["scenario"], model_name=run["model_name"], provider=run["provider"], threshold=float(run["threshold"]), dataset_summary=run.get("dataset_summary", {}), metrics=run.get("metrics", {}), errors=[], latency=run.get("hardware", {}), license_status=license_status, created_at=run.get("created_at") or utc_now_iso())
        paths = write_reports(result, self.context.config.report_dir, language=self.context.config.ui_language)
        report_path = paths.get("pdf") or paths["markdown"]
        self.context.storage.save_evaluation_run(result.scenario, result.model_name, result.provider, result.threshold, result.dataset_summary, result.metrics, result.latency, report_path, result.created_at)
        self.refresh()
        self.set_status(f"Report re-exported to {report_path}")

    def delete_selected(self) -> None:
        run = self._selected_run()
        if not run:
            return
        self.context.storage.delete_evaluation_run(int(run["id"]))
        self.refresh()

    def export_summary(self) -> None:
        path = Path(self.context.config.export_dir) / f"evaluation_summary_{timestamp_for_filename()}.csv"
        export_csv(path, self.runs)
        self.set_status(f"Evaluation summary exported to {path}")
