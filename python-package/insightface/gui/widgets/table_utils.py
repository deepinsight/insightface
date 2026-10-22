"""Table layout helpers for dense desktop review pages."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QTimer
from PySide6.QtWidgets import QAbstractItemView, QAbstractScrollArea, QHeaderView, QTableWidget


class _ProportionalTableSizer(QObject):
    def __init__(self, table: QTableWidget, estimated_widths: list[int], minimum_width: int = 56):
        super().__init__(table)
        self.table = table
        self.estimated_widths = [max(1, int(width)) for width in estimated_widths]
        self.minimum_width = max(1, int(minimum_width))

    def update(self, estimated_widths: list[int], minimum_width: int) -> None:
        self.estimated_widths = [max(1, int(width)) for width in estimated_widths]
        self.minimum_width = max(1, int(minimum_width))

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if event.type() == QEvent.Resize:
            QTimer.singleShot(0, self.apply)
        return False

    def apply(self) -> None:
        columns = self.table.columnCount()
        if columns <= 0:
            return
        estimates = self.estimated_widths[:columns]
        if len(estimates) < columns:
            estimates.extend([self.minimum_width] * (columns - len(estimates)))
        available = max(1, self.table.viewport().width())
        total = max(1, sum(estimates))
        used = 0
        for column in range(columns):
            if column == columns - 1:
                width = max(self.minimum_width, available - used)
            else:
                width = max(self.minimum_width, int(available * estimates[column] / total))
                used += width
            self.table.setColumnWidth(column, width)


def configure_table_columns(table: QTableWidget, estimated_widths: list[int], minimum_width: int = 56) -> None:
    """Make table columns fill available width using estimated relative widths."""

    existing = getattr(table, "_proportional_table_sizer", None)
    if existing is not None:
        existing.update(estimated_widths, minimum_width)
        existing.apply()
        return

    table.setAlternatingRowColors(True)
    table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
    table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustIgnored)
    header = table.horizontalHeader()
    header.setStretchLastSection(False)
    header.setMinimumSectionSize(minimum_width)
    header.setSectionResizeMode(QHeaderView.Interactive)
    sizer = _ProportionalTableSizer(table, estimated_widths, minimum_width)
    table.installEventFilter(sizer)
    table.viewport().installEventFilter(sizer)
    table._proportional_table_sizer = sizer  # type: ignore[attr-defined]
    QTimer.singleShot(0, sizer.apply)


def refresh_table_columns(table: QTableWidget) -> None:
    """Re-apply proportional sizing after table data changes."""

    sizer = getattr(table, "_proportional_table_sizer", None)
    if sizer is not None:
        sizer.apply()
    else:
        table.resizeColumnsToContents()
