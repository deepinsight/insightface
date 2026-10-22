"""Table helpers for face/search results."""

from __future__ import annotations

from typing import Iterable, Mapping

from PySide6.QtWidgets import QTableWidget, QTableWidgetItem

from .table_utils import configure_table_columns, refresh_table_columns


class FaceTable(QTableWidget):
    def set_rows(self, rows: Iterable[Mapping[str, object]], columns: list[str]) -> None:
        rows = list(rows)
        self.setColumnCount(len(columns))
        self.setHorizontalHeaderLabels(columns)
        configure_table_columns(self, [140] * len(columns))
        self.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for col_index, column in enumerate(columns):
                self.setItem(row_index, col_index, QTableWidgetItem(str(row.get(column, ""))))
        refresh_table_columns(self)
