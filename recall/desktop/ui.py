"""Small PySide6 desktop shell for the first RE-call workflow."""

from __future__ import annotations

import importlib
from dataclasses import replace
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from recall.desktop.models import RuntimeMode, RuntimeProfile, SourceCategory, SourceSelection
from recall.desktop.profiles import load_pipelines, save_pipelines, save_profile
from recall.desktop.runtime import (
    _CORPUS_SUFFIXES,
    DockerRuntime,
    RuntimeManager,
    RuntimeErrorBase,
    create_runtime,
)
from recall.desktop.sources import (
    CLAUDE_MEMORY_FILENAMES,
    CODE_EXTENSIONS,
    DOCUMENT_EXTENSIONS,
    collect_files,
    default_scan_roots,
    scan_files,
    classify,
    display_type,
)
from recall.desktop.github import GithubImport, download_repository
from recall.desktop.jobs import CLOSE_WAIT_MS
from recall.desktop.jobs import Job as _Worker
from recall.store import scrub_dsn_secrets
from recall.wizard.database import probe_database


_qt_widgets: Any = None
try:
    _qt_widgets = importlib.import_module("PySide6.QtWidgets")
    from PySide6.QtCore import QEvent, QItemSelectionModel, QObject, QPoint, QThreadPool, QTimer, Qt, Signal
    from PySide6.QtGui import QColor, QPixmap, QPolygon
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QComboBox,
        QCheckBox,
        QFileDialog,
        QFrame,
        QFormLayout,
        QGraphicsOpacityEffect,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QInputDialog,
        QGridLayout,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QGroupBox,
        QLineEdit,
        QSizePolicy,
        QStyle,
        QStyleOptionHeader,
        QStyledItemDelegate,
        QStackedWidget,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except ImportError:  # pragma: no cover, exercised on environments without the desktop extra
    pass

QApplication: Any = getattr(_qt_widgets, "QApplication", None)


if QApplication is not None:

    def _project_name(value: str) -> str:
        """Show one project entry for a project's corpus scopes.

        ⚠️ **This listed `("-docs", "-code")` and missed `-memory`** — the same omission, in the
        same words, that `runtime.py` records having made in three places at once. It survived here
        because nothing pinned the two files together. With memory tenants now real, a runtime that
        returns raw tenant names (the base `RuntimeManager.list_tenants` does, from `recall_tenants`)
        yields a phantom project called `<project>-memory`, and the calibration page then builds
        `<project>-memory-docs` from it.
        """
        text = value.strip()
        for suffix in _CORPUS_SUFFIXES:
            if text.endswith(suffix):
                return text[: -len(suffix)]
        return text


    def _format_summary() -> str:
        suffixes = sorted(
            suffix.removeprefix(".").upper()
            for suffix in DOCUMENT_EXTENSIONS | CODE_EXTENSIONS
        )
        midpoint = (len(suffixes) + 1) // 2
        first_line = ", ".join(suffixes[:midpoint])
        second_line = ", ".join(suffixes[midpoint:])
        return f"Available formats: {first_line}\n{second_line}"

    class _DropZone(QFrame):
        dropped = Signal(list)

        def __init__(self) -> None:
            super().__init__()
            self.setAcceptDrops(True)
            self.setObjectName("dropZone")
            layout = QVBoxLayout(self)
            layout.setSpacing(4)
            label = QLabel("DROP FILES HERE")
            label.setObjectName("dropTitle")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            layout.addWidget(label)
            hint = QLabel(_format_summary())
            hint.setObjectName("dropHint")
            hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
            hint.setWordWrap(False)
            hint.setMinimumWidth(0)
            hint.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            hint.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            layout.addWidget(hint)
            note = QLabel("Unsupported files will be skipped")
            note.setObjectName("dropNote")
            note.setAlignment(Qt.AlignmentFlag.AlignCenter)
            note.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            layout.addWidget(note)
            self.setMinimumHeight(180)

        def dragEnterEvent(self, event: Any) -> None:
            if event.mimeData().hasUrls():
                self.setProperty("dragActive", True)
                self.style().unpolish(self)
                self.style().polish(self)
                event.acceptProposedAction()

        def dragMoveEvent(self, event: Any) -> None:
            if event.mimeData().hasUrls():
                event.acceptProposedAction()

        def dropEvent(self, event: Any) -> None:
            paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
            if paths:
                self.dropped.emit(paths)
            self._clear_drag_state()
            event.acceptProposedAction()

        def dragLeaveEvent(self, event: Any) -> None:
            self._clear_drag_state()
            event.accept()

        def _clear_drag_state(self) -> None:
            self.setProperty("dragActive", False)
            self.style().unpolish(self)
            self.style().polish(self)


    class _FocuslessItemDelegate(QStyledItemDelegate):
        """Keep selected table rows highlighted without drawing Qt focus boxes."""

        def __init__(self, parent: QWidget | None = None, *, hide_selection: bool = False) -> None:
            super().__init__(parent)
            self._hide_selection = hide_selection

        def paint(self, painter: Any, option: Any, index: Any) -> None:
            option.state &= ~QStyle.StateFlag.State_HasFocus
            if self._hide_selection:
                option.state &= ~QStyle.StateFlag.State_Selected
            super().paint(painter, option, index)


    class _FileHeader(QHeaderView):
        """Keep the descending marker native-sized while placing it below the label."""

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(Qt.Orientation.Horizontal, parent)
            self._descending_section = -1
            self.setSectionsClickable(True)
            self.setSortIndicatorShown(True)

        def setDescendingSection(self, section: int) -> None:
            self._descending_section = section
            self.viewport().update()

        def paintSection(self, painter: Any, rect: Any, logical_index: int) -> None:
            super().paintSection(painter, rect, logical_index)
            if logical_index != self._descending_section:
                return

            option = QStyleOptionHeader()
            option.rect = rect
            option.section = logical_index
            option.orientation = Qt.Orientation.Horizontal
            option.sortIndicator = QStyleOptionHeader.SortIndicator.SortDown
            mark_size = self.style().pixelMetric(QStyle.PixelMetric.PM_HeaderMarkSize, option, self)
            mark_size = max(6, mark_size)
            y = rect.bottom() - mark_size - 4
            center_x = rect.center().x()
            half = max(3, mark_size // 2)
            painter.save()
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#d7d2c4"))
            painter.drawPolygon(
                QPolygon(
                    [
                        QPoint(center_x - half, y),
                        QPoint(center_x + half, y),
                        QPoint(center_x, y + half),
                    ]
                )
            )
            painter.restore()


    class _HiddenSortItem(QTableWidgetItem):
        """Keep a sortable value without painting a duplicate behind a cell widget."""

        _SORT_ROLE = Qt.ItemDataRole.UserRole + 1

        def __init__(self, sort_value: str) -> None:
            super().__init__("")
            self.setData(self._SORT_ROLE, sort_value.casefold())

        def __lt__(self, other: QTableWidgetItem) -> bool:
            left = str(self.data(self._SORT_ROLE) or "")
            right = str(other.data(self._SORT_ROLE) or "")
            return left < right


    class _TableWatermark(QLabel):
        """Scale a table watermark to the available cell without cropping it."""

        def __init__(self, source: QPixmap) -> None:
            super().__init__()
            self._source = source
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)

        def resizeEvent(self, event: Any) -> None:
            available = min(max(0, self.width() - 40), max(0, self.height() - 40), 420)
            if available > 0:
                self.setPixmap(
                    self._source.scaled(
                        available,
                        available,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
            super().resizeEvent(event)


    class _PageWatermark(_TableWatermark):
        """Keep a centered watermark fitted to a full tab page."""

        def __init__(self, page: QWidget, source: QPixmap) -> None:
            super().__init__(source)
            self._page = page
            page.installEventFilter(self)
            self.setGeometry(page.rect())

        def eventFilter(self, watched: QObject, event: QEvent) -> bool:
            if watched is self._page and event.type() == QEvent.Type.Resize:
                self.setGeometry(self._page.rect())
            return bool(super().eventFilter(watched, event))


    # ⛔ **Moved to `recall/desktop/jobs.py`, not copied.** The graphical installer needs the same
    # runnable with a progress channel, and the two properties that make a queued cross-thread
    # signal actually arrive are a race when they are wrong — they work often enough to look
    # correct, so a second copy would not fail visibly. The measurements and the reasoning live in
    # that module's docstring; `_run` below still explains why it holds the reference, because that
    # is where a reader of this file will look.
    _CLOSE_WAIT_MS = CLOSE_WAIT_MS


    class MainWindow(QMainWindow):
        def __init__(self, profile: RuntimeProfile, runtime: RuntimeManager | None = None) -> None:
            super().__init__()
            self.profile = profile
            self.runtime = runtime or create_runtime(profile)
            self.pool = QThreadPool(self)
            #: Live workers. Holding them is what makes `_run`'s callbacks arrive at all; see the
            #: measurement in `_run`. Entries are removed when their job reports.
            self._workers: list[Any] = []
            self.pending_files: list[tuple[Path, SourceCategory]] = []
            self.pending_scopes: list[dict[str, Any]] = []
            self.calibration_snapshot: Any = None
            self._project_names: list[str] = [self.profile.default_tenant]
            self._last_scope_index = 0
            self._config_dirty = False
            self._calibration_required = False
            self._calibration_targets_by_row: list[tuple[str, str, str]] = []
            self._calibration_results: dict[str, Any] = {}
            self._latest_release: Any = None
            self._calibration_running = False
            self._api_keys: dict[str, str] = {"OpenRouter": "", "Voyage": "", "OpenAI": ""}
            self.github_pending: list[tuple[Path, SourceCategory]] = []
            self.github_root: Path | None = None
            self.github_import: GithubImport | None = None
            self._active_config_type = "Documents"
            self._pipeline_configs: dict[str, dict[str, Any]] = {
                source_type: {
                    "embedder": "Hashing offline",
                    "reranker": "Disabled",
                    "splade": False,
                    "judge": "Disabled",
                    "reasoning": "Disabled",
                    "model": "hashing-64",
                    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    "arm": "Threshold",
                }
                for source_type in ("Documents", "Memory", "Code")
            }
            # Saved choices layered OVER the defaults, key by key. A wholesale replacement would
            # drop any setting added after the file was written, so an upgrade would silently lose
            # a control's value rather than keep the new default for it.
            for source_type, saved in load_pipelines().items():
                if source_type in self._pipeline_configs:
                    self._pipeline_configs[source_type].update(saved)
            self.setWindowTitle("RE-call")
            self.resize(980, 760)
            self._build_ui()

        def _add_table_watermark(self, table: QTableWidget, stack: QGridLayout) -> None:
            """Place a quiet centered RE-call watermark behind a table's content."""
            table.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            table.viewport().setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            table.setAutoFillBackground(False)
            table.viewport().setAutoFillBackground(False)
            watermark_path = Path(__file__).with_name("assets") / "re_call_watermark.png"
            pixmap = QPixmap(str(watermark_path))
            if pixmap.isNull():
                return
            watermark = _TableWatermark(pixmap)
            watermark.setObjectName("tableWatermark")
            watermark.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            effect = QGraphicsOpacityEffect(watermark)
            effect.setOpacity(0.11)
            watermark.setGraphicsEffect(effect)
            stack.addWidget(watermark, 0, 0)

        def _add_page_watermark(self, page: QWidget) -> None:
            """Place the same restrained watermark behind a settings-style page."""
            watermark_path = Path(__file__).with_name("assets") / "re_call_watermark.png"
            pixmap = QPixmap(str(watermark_path))
            if pixmap.isNull():
                return
            watermark = _PageWatermark(page, pixmap)
            watermark.setObjectName("tableWatermark")
            watermark.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            effect = QGraphicsOpacityEffect(watermark)
            effect.setOpacity(0.08)
            watermark.setGraphicsEffect(effect)
            watermark.lower()
            watermark.show()

        def _build_ui(self) -> None:
            self.setStyleSheet(
                """
                QMainWindow, QWidget { background: #0e100f; color: #f4f1e8; font-family: "Segoe UI Variable Text", "Segoe UI"; font-size: 13px; }
                QLabel { color: #b6b7ac; }
                QLabel#tableWatermark { background: transparent; }
                QLabel#runtimeLabel { color: #b6b7ac; font-family: "Consolas"; font-size: 11px; }
                QLabel#status { color: #b6b7ac; font-size: 12px; }
                QFrame#dropZone { background: #141714; border: 1px dashed #465047; border-radius: 4px; padding: 18px; }
                QFrame#dropZone:hover, QFrame#dropZone[dragActive="true"] { border-color: #d7a52a; background: #191b15; }
                QFrame#dropZone QLabel { background: transparent; }
                QLabel#dropTitle { color: #f4f1e8; font-family: "Segoe UI Variable Display", "Segoe UI"; font-size: 24px; font-weight: 700; letter-spacing: 0.4px; }
                QLabel#dropHint { color: #b6b7ac; font-size: 10px; line-height: 1.2; }
                QLabel#dropNote { color: #8d9186; font-size: 12px; }
                QPushButton { background: #171a17; color: #f4f1e8; border: 1px solid #465047; border-radius: 3px; padding: 0px 16px; min-height: 40px; font-family: "Segoe UI Variable Text", "Segoe UI"; font-size: 13px; font-weight: 600; }
                QPushButton:hover { background: #20241f; border-color: #d7a52a; }
                QPushButton:pressed { background: #2a2415; border-color: #f0be4a; }
                QPushButton:disabled { background: #141614; color: #70766d; border-color: #2c322c; }
                QPushButton:focus, QComboBox:focus, QLineEdit:focus, QCheckBox:focus { outline: none; border: 1px solid #f0be4a; }
                QPushButton#navButton { background: #131613; color: #d9d6cc; border-color: #465047; border-radius: 3px; padding: 0px; min-height: 34px; max-height: 34px; font-family: "Segoe UI Variable Text", "Segoe UI"; font-size: 12px; font-weight: 700; letter-spacing: 0.4px; }
                QPushButton#navButton:hover { background: rgba(215, 165, 42, 0.10); border-color: #d7a52a; color: #f4f1e8; }
                QPushButton#navButton:checked, QPushButton#downloadButton { background: rgba(215, 165, 42, 0.22); border-color: #d7a52a; color: #f4f1e8; min-height: 34px; max-height: 34px; padding: 0px; }
                QPushButton#navButton:checked:hover, QPushButton#downloadButton:hover { background: rgba(215, 165, 42, 0.32); border-color: #f0be4a; color: #ffffff; }
                QPushButton#downloadButton:pressed { background: rgba(215, 165, 42, 0.42); border-color: #f0be4a; color: #ffffff; }
                QPushButton#downloadButton:disabled { background: rgba(215, 165, 42, 0.08); border-color: #5f4d25; color: #8b7850; }
                QPushButton#githubSecondaryButton { min-height: 34px; max-height: 34px; padding: 0px; }
                QPushButton#startButton { background: #d7a52a; color: #11130f; border-color: #d7a52a; padding: 0px 18px; min-height: 40px; font-weight: 700; }
                QPushButton#startButton:hover { background: #f0be4a; border-color: #f0be4a; }
                QPushButton#startButton:pressed { background: #b8871d; border-color: #b8871d; }
                QPushButton#startButton:disabled { background: #2b2618; color: #8b7850; border-color: #5f4d25; }
                QComboBox { background: #141714; color: #f4f1e8; border: 1px solid #465047; border-radius: 3px; padding: 0px 12px; min-height: 40px; min-width: 120px; }
                QComboBox:hover, QComboBox:focus { border-color: #d7a52a; }
                QComboBox#tenantCellCombo { background: transparent; border: 0; border-radius: 0; padding: 2px 6px; min-width: 0; }
                QComboBox#tenantCellCombo:hover, QComboBox#tenantCellCombo:focus { background: transparent; border: 0; outline: none; }
                QComboBox#tenantCellCombo::drop-down { background: transparent; border: 0; width: 18px; }
                QComboBox QAbstractItemView { background: #1a1d1a; color: #f4f1e8; selection-background-color: #d7a52a; selection-color: #11130f; border: 1px solid #465047; }
                QTableWidget { background: #111411; color: #f4f1e8; border: 1px solid #465047; border-radius: 4px; gridline-color: #2a2f2a; selection-background-color: #3a301b; selection-color: #f4f1e8; padding: 4px; font-size: 13px; }
                QTableWidget::item { color: #f4f1e8; padding: 9px 10px; border: 0; }
                QTableWidget::item:selected { background: #3a301b; color: #f4f1e8; }
                QTableWidget#calibrationTable { selection-background-color: transparent; }
                QTableWidget#calibrationTable::item:selected, QTableWidget#calibrationTable::item:focus { background: transparent; border: 0; outline: none; }
                QTableWidget#filesTable, QTableWidget#githubTable, QTableWidget#calibrationTable { background: transparent; }
                QTableWidget#filesTable::item, QTableWidget#githubTable::item, QTableWidget#calibrationTable::item { background: transparent; }
                QHeaderView::section { background: #1a1d1a; color: #d7d2c4; font-family: "Consolas"; font-size: 11px; font-weight: 700; letter-spacing: 1px; padding: 10px; border: 0; }
                QTableWidget#filesTable QHeaderView, QTableWidget#githubTable QHeaderView { background: transparent; }
                QTableWidget#filesTable QHeaderView::section, QTableWidget#githubTable QHeaderView::section { background: transparent; }
                QTableCornerButton::section { background: transparent; border: 0; }
                QFrame#queueActions { background: #151815; border: 1px solid #465047; border-radius: 3px; }
                QFrame#queueActions QPushButton { padding: 0px 14px; }
                QWidget#calibrationActionsCell { background: transparent; }
                QPushButton#tableActionButton { padding: 0px 10px; min-height: 32px; font-size: 12px; }
                QPushButton#tableActionButton:focus { outline: none; border: 1px solid #f0be4a; background: #20241f; }
                QProgressBar { background: #141714; color: #f4f1e8; border: 1px solid #465047; border-radius: 3px; height: 18px; text-align: center; }
                QProgressBar::chunk { background: #d7a52a; border-radius: 2px; }
                QScrollBar:vertical { background: #111411; width: 12px; margin: 0; }
                QScrollBar::handle:vertical { background: #465047; border-radius: 2px; min-height: 28px; }
                QScrollBar::handle:vertical:hover { background: #697267; }
                QFrame#runtimeStatus { background: #151815; border: 1px solid #465047; border-radius: 3px; padding: 5px 10px; }
                QFrame#runtimeStatus QLabel#runtimeLabel { background: transparent; }
                QPushButton#reconnectButton { color: #f0be4a; border-color: #8e6c20; padding: 0px 14px; }
                QLabel#connectionLight { min-width: 10px; max-width: 10px; min-height: 10px; max-height: 10px; border-radius: 5px; }
                QGroupBox { color: #f4f1e8; border: 1px solid #465047; border-radius: 4px; margin-top: 14px; padding: 16px; }
                QGroupBox#watermarkGroup { background: transparent; }
                QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 7px; color: #d7a52a; font-size: 12px; font-weight: 700; }
                QLineEdit { background: #141714; color: #f4f1e8; border: 1px solid #465047; border-radius: 3px; padding: 0px 12px; min-height: 40px; }
                QLineEdit::placeholder { color: #8d9186; }
                QCheckBox { color: #f4f1e8; spacing: 8px; min-height: 32px; }
                QCheckBox::indicator { width: 17px; height: 17px; }
                QTableWidget#calibrationTable { border-radius: 4px; }
                QLabel#pageTitle { color: #f4f1e8; font-family: "Segoe UI Variable Display", "Segoe UI"; font-size: 24px; font-weight: 700; letter-spacing: 0.3px; }
                QLabel#pageIntro { color: #b6b7ac; font-size: 13px; }
                QLabel#warningLabel { color: #d9b35a; background: #211b0e; border: 1px solid #80621c; border-radius: 3px; padding: 12px; }
                QLabel#successLabel { color: #63d39e; }
                QLabel#mutedLabel { color: #8d9186; }
                """
            )
            root = QWidget()
            outer = QVBoxLayout(root)
            outer.setContentsMargins(16, 12, 16, 12)
            outer.setSpacing(12)
            header = QHBoxLayout()
            header.setSpacing(8)
            header.addStretch()
            self.main_button = self._make_nav_button("MAIN", 0)
            self.github_button = self._make_nav_button("GITHUB", 1)
            self.calibration_button = self._make_nav_button("CALIBRATION", 2)
            self.config_button = self._make_nav_button("CONFIG", 3)
            self.settings_button = self._make_nav_button("SETTINGS", 4)
            header.addWidget(self.main_button)
            header.addWidget(self.github_button)
            header.addWidget(self.calibration_button)
            header.addWidget(self.config_button)
            header.addWidget(self.settings_button)
            outer.addLayout(header)

            self.pages = QStackedWidget()
            self.pages.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
            self.queue_page = QWidget()
            queue_outer = QVBoxLayout(self.queue_page)
            queue_outer.setContentsMargins(0, 0, 0, 0)
            queue_outer.setSpacing(10)
            self.pages.addWidget(self.queue_page)
            outer.addWidget(self.pages, 1)

            controls = QHBoxLayout()
            controls.setSpacing(10)
            controls.addWidget(QLabel("Default project"))
            self.scope = QComboBox()
            self.scope.setMinimumWidth(230)
            self._populate_scopes(self._project_names)
            self.scope.currentIndexChanged.connect(self._scope_changed)
            controls.addWidget(self.scope)
            controls.addWidget(QLabel("New files use this project; adjust tenants per row"))
            self.start_button = QPushButton("Start indexing")
            self.start_button.setObjectName("startButton")
            self.start_button.setEnabled(False)
            self.start_button.clicked.connect(self._start_embedding)
            controls.addStretch()
            queue_outer.addLayout(controls)

            zone = _DropZone()
            zone.dropped.connect(self._accept_drop)
            queue_outer.addWidget(zone)

            self.files = QTableWidget(0, 3)
            self.files.setObjectName("filesTable")
            self.files.setHorizontalHeaderLabels(["FILE NAME", "TENANT", "TYPE"])
            self.files.setHorizontalHeader(_FileHeader(self.files))
            self.files.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
            self.files.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
            self.files.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
            self.files.setColumnWidth(1, 210)
            self.files.setColumnWidth(2, 78)
            self.files.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.files.setSelectionMode(QAbstractItemView.ExtendedSelection)
            self.files.setItemDelegate(_FocuslessItemDelegate(self.files))
            self.files.setEditTriggers(QAbstractItemView.NoEditTriggers)
            self.files.verticalHeader().setVisible(False)
            self.files.verticalHeader().setDefaultSectionSize(42)
            self.files.setShowGrid(False)
            self.files.setCornerButtonEnabled(False)
            self.files.setSortingEnabled(True)
            self.files.horizontalHeader().setSortIndicatorShown(True)
            self.files.horizontalHeader().setMinimumHeight(46)
            self.files.horizontalHeader().sortIndicatorChanged.connect(self._update_file_sort_indicator)
            self._update_file_sort_indicator(
                self.files.horizontalHeader().sortIndicatorSection(),
                self.files.horizontalHeader().sortIndicatorOrder(),
            )
            table_frame = QWidget()
            table_stack = QGridLayout(table_frame)
            table_stack.setContentsMargins(0, 0, 0, 0)
            table_stack.setSpacing(0)
            self._add_table_watermark(self.files, table_stack)
            table_stack.addWidget(self.files, 0, 0)
            actions = QFrame()
            actions.setObjectName("queueActions")
            action_layout = QHBoxLayout(actions)
            action_layout.setContentsMargins(6, 5, 6, 5)
            action_layout.setSpacing(6)
            browse = QPushButton("Browse")
            browse.clicked.connect(self._browse)
            clear = QPushButton("Clear")
            clear.clicked.connect(self._clear_files)
            self.cancel_button = QPushButton("Cancel")
            self.cancel_button.clicked.connect(self._cancel_selected_files)
            action_layout.addWidget(browse)
            action_layout.addWidget(clear)
            action_layout.addWidget(self.cancel_button)
            table_stack.addWidget(
                actions,
                0,
                0,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
            )
            start_holder = QFrame()
            start_holder.setObjectName("queueActions")
            start_layout = QHBoxLayout(start_holder)
            start_layout.setContentsMargins(6, 5, 6, 5)
            self.scan_button = QPushButton("Scan")
            self.scan_button.setToolTip(
                "Scan Documents, Desktop, Downloads, and Claude memory folders for new sources"
            )
            self.scan_button.clicked.connect(self._scan_requested)
            start_layout.addWidget(self.scan_button)
            start_layout.addWidget(self.start_button)
            table_stack.addWidget(
                start_holder,
                0,
                0,
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
            )
            queue_outer.addWidget(table_frame, 1)
            self.progress = QProgressBar()
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            queue_outer.addWidget(self.progress)

            footer = QHBoxLayout()
            self.status = QLabel("Connect to a runtime to begin.")
            # PlainText, not Qt's default AutoText: this label reports user-supplied names (a
            # project typed into "+ Add project") and error text, and AutoText renders anything
            # that looks like markup AS markup. A status line that can be visually rewritten by
            # its own subject is worth nothing, and this is the line that says "not provisioned".
            self.status.setTextFormat(Qt.TextFormat.PlainText)
            self.status.setObjectName("status")
            footer.addWidget(self.status, 1)
            self.reconnect_button = QPushButton("Reconnect")
            self.reconnect_button.setObjectName("reconnectButton")
            self.reconnect_button.clicked.connect(self._reconnect)
            self.reconnect_button.setVisible(True)
            footer.addWidget(self.reconnect_button)
            runtime_status = QFrame()
            runtime_status.setObjectName("runtimeStatus")
            runtime_status.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            runtime_layout = QHBoxLayout(runtime_status)
            runtime_layout.setContentsMargins(8, 4, 8, 4)
            self.connection_light = QLabel()
            self.connection_light.setObjectName("connectionLight")
            runtime_layout.addWidget(self.connection_light)
            self.runtime_label = QLabel()
            self.runtime_label.setObjectName("runtimeLabel")
            self.runtime_label.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            runtime_layout.addWidget(self.runtime_label)
            footer.addWidget(runtime_status)
            queue_outer.addLayout(footer)
            self._set_runtime_state(False)
            self.github_page = self._build_github_page()
            self.calibration_page = self._build_calibration_page()
            self.config_page = self._build_config_page()
            self.settings_page = self._build_settings_page()
            self.pages.addWidget(self.github_page)
            self.pages.addWidget(self.calibration_page)
            self.pages.addWidget(self.config_page)
            self.pages.addWidget(self.settings_page)
            self.setCentralWidget(root)
            self._run(self._prepare_runtime, self._runtime_ready, self._runtime_failed)
            self.update_timer = QTimer(self)
            self.update_timer.setInterval(6 * 60 * 60 * 1000)
            self.update_timer.timeout.connect(lambda: self._run(self._check_update_safely, self._update_checked))
            self.update_timer.start()

        def _make_nav_button(self, label: str, page_index: int) -> QPushButton:
            button = QPushButton(label)
            button.setObjectName("navButton")
            button.setCheckable(True)
            button.setFixedSize(112, 34)
            button.clicked.connect(lambda _checked=False: self._show_page(page_index))
            return button

        def _show_page(self, page_index: int) -> None:
            self.pages.setCurrentIndex(page_index)
            for index, button in (
                (0, self.main_button),
                (1, self.github_button),
                (2, self.calibration_button),
                (3, self.config_button),
                (4, self.settings_button),
            ):
                button.setChecked(index == page_index)
            if page_index == 2:
                self._refresh_calibration_table()

        def _build_github_page(self) -> QWidget:
            page = QWidget()
            layout = QVBoxLayout(page)
            title = QLabel("GITHUB")
            title.setObjectName("pageTitle")
            layout.addWidget(title)
            intro = QLabel(
                "Download a public repository into a local review queue. Choose what to import, "
                "approve the files, then start indexing from Main."
            )
            intro.setObjectName("pageIntro")
            intro.setWordWrap(True)
            layout.addWidget(intro)

            source_group = QGroupBox("Repository source")
            source_form = QFormLayout(source_group)
            self.github_url_edit = QLineEdit()
            self.github_url_edit.setPlaceholderText("https://github.com/owner/repository")
            self.github_scope_combo = QComboBox()
            self.github_scope_combo.addItems(["Full repository", "Code only", "Documents only"])
            self.github_tenant_combo = QComboBox()
            self._populate_github_tenants(self._project_names)
            source_form.addRow("Repository URL", self.github_url_edit)
            source_form.addRow("Import scope", self.github_scope_combo)
            source_form.addRow("Project", self.github_tenant_combo)
            layout.addWidget(source_group)

            self.github_table = QTableWidget(0, 3)
            self.github_table.setObjectName("githubTable")
            self.github_table.setHorizontalHeaderLabels(["FILE NAME", "TENANT", "TYPE"])
            self.github_table.setHorizontalHeader(_FileHeader(self.github_table))
            self.github_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
            self.github_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
            self.github_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
            self.github_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.github_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
            self.github_table.setItemDelegate(_FocuslessItemDelegate(self.github_table))
            self.github_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            self.github_table.verticalHeader().setVisible(False)
            self.github_table.verticalHeader().setDefaultSectionSize(36)
            self.github_table.setShowGrid(False)
            self.github_table.setCornerButtonEnabled(False)
            github_table_frame = QWidget()
            github_table_stack = QGridLayout(github_table_frame)
            github_table_stack.setContentsMargins(0, 0, 0, 0)
            github_table_stack.setSpacing(0)
            self._add_table_watermark(self.github_table, github_table_stack)
            github_table_stack.addWidget(self.github_table, 0, 0)
            layout.addWidget(github_table_frame, 1)

            actions = QHBoxLayout()
            self.github_download_button = QPushButton("DOWNLOAD")
            self.github_download_button.setObjectName("downloadButton")
            self.github_download_button.setFixedSize(112, 34)
            self.github_download_button.clicked.connect(self._download_github)
            self.github_clear_button = QPushButton("Clear")
            self.github_clear_button.setObjectName("githubSecondaryButton")
            self.github_clear_button.setFixedSize(112, 34)
            self.github_clear_button.clicked.connect(self._clear_github)
            self.github_approve_button = QPushButton("Approve files and open queue")
            self.github_approve_button.setEnabled(False)
            self.github_approve_button.clicked.connect(self._approve_github)
            actions.addWidget(self.github_download_button)
            actions.addWidget(self.github_clear_button)
            actions.addStretch()
            actions.addWidget(self.github_approve_button)
            layout.addLayout(actions)

            self.github_status = QLabel("Enter a repository URL to begin.")
            self.github_status.setObjectName("status")
            self.github_status.setWordWrap(True)
            layout.addWidget(self.github_status)
            return page

        def _build_config_page(self) -> QWidget:
            page = QWidget()
            layout = QVBoxLayout(page)
            title = QLabel("PIPELINE CONFIGURATION")
            title.setObjectName("pageTitle")
            layout.addWidget(title)
            intro = QLabel(
                "Choose the retrieval components RE-call should use for new indexing and search jobs. "
                "These choices are runtime-wide and must remain compatible with the detected machine."
            )
            intro.setObjectName("pageIntro")
            intro.setWordWrap(True)
            layout.addWidget(intro)

            type_row = QHBoxLayout()
            type_row.addWidget(QLabel("Configuration for file type"))
            self.config_type_combo = QComboBox()
            self.config_type_combo.addItems(["Documents", "Memory", "Code"])
            self.config_type_combo.currentTextChanged.connect(self._switch_config_type)
            type_row.addWidget(self.config_type_combo)
            type_row.addStretch()
            layout.addLayout(type_row)

            pipeline = QGroupBox("Retrieval stack")
            pipeline.setObjectName("watermarkGroup")
            form = QFormLayout(pipeline)
            self.embedder_combo = QComboBox()
            self.embedder_combo.addItems(
                [
                    "Hashing offline",
                    "Sentence Transformers local",
                    "SFR code local",
                    "OpenAI-compatible cloud",
                    "Voyage cloud (unavailable)",
                    "BGE local (unavailable)",
                ]
            )
            self._disable_config_items(
                self.embedder_combo,
                {"Voyage cloud (unavailable)", "BGE local (unavailable)"},
            )
            self.reranker_combo = QComboBox()
            self.reranker_combo.addItems(
                [
                    "Disabled",
                    "Local cross-encoder",
                    "Local code reranker",
                    "Local model path",
                    "Voyage reranker (unavailable)",
                    "BGE reranker (unavailable)",
                ]
            )
            self._disable_config_items(
                self.reranker_combo,
                {"Voyage reranker (unavailable)", "BGE reranker (unavailable)"},
            )
            self.splade_check = QCheckBox("Enable learned sparse SPLADE retrieval")
            self.splade_check.setEnabled(False)
            self.splade_status = QLabel("Checking CUDA and VRAM capability…")
            self.splade_status.setObjectName("mutedLabel")
            splade_row = QWidget()
            splade_layout = QHBoxLayout(splade_row)
            splade_layout.setContentsMargins(0, 0, 0, 0)
            splade_layout.addWidget(self.splade_check)
            splade_layout.addWidget(self.splade_status, 1)
            self.judge_combo = QComboBox()
            self.judge_combo.addItems(["Disabled", "Local entailment judge", "Remote LLM judge"])
            self.reasoning_combo = QComboBox()
            self.reasoning_combo.addItems(["Disabled", "Enabled"])
            self.model_edit = QLineEdit("hashing-64")
            self.model_edit.setPlaceholderText("Embedder model identifier or pinned alias")
            self.reranker_model_edit = QLineEdit("cross-encoder/ms-marco-MiniLM-L-6-v2")
            self.reranker_model_edit.setPlaceholderText("Local reranker model id or artifact path")
            self.arm_combo = QComboBox()
            self.arm_combo.addItems(["Threshold", "Stacked", "Entail only"])
            form.addRow("Embedder", self.embedder_combo)
            form.addRow("Re-ranker", self.reranker_combo)
            form.addRow("SPLADE", splade_row)
            form.addRow("Judge", self.judge_combo)
            form.addRow("Reasoning", self.reasoning_combo)
            form.addRow("Embedder model", self.model_edit)
            form.addRow("Reranker model/path", self.reranker_model_edit)
            form.addRow("Calibration arm", self.arm_combo)
            layout.addWidget(pipeline)

            self.configuration_warning = QLabel(
                "BGE and Voyage remain visible as unavailable options. Any new configuration set requires a complete corpus calibration before it can be activated."
            )
            self.configuration_warning.setObjectName("warningLabel")
            self.configuration_warning.setWordWrap(True)
            layout.addWidget(self.configuration_warning)
            config_actions = QHBoxLayout()
            config_actions.addStretch()
            save = QPushButton("Save configuration")
            save.clicked.connect(self._save_configuration)
            config_actions.addWidget(save)
            layout.addLayout(config_actions)
            layout.addStretch()

            for control in (
                self.embedder_combo,
                self.reranker_combo,
                self.judge_combo,
                self.reasoning_combo,
                self.arm_combo,
            ):
                control.currentTextChanged.connect(self._configuration_changed)
            self.splade_check.stateChanged.connect(self._configuration_changed)
            self.model_edit.textChanged.connect(self._configuration_changed)
            self.reranker_model_edit.textChanged.connect(self._configuration_changed)
            self._load_config_type(self._active_config_type)
            self._run(self._probe_hardware, self._hardware_probe_done, self._hardware_probe_failed)
            self._add_page_watermark(page)
            return page

        def _disable_config_items(self, combo: QComboBox, labels: set[str]) -> None:
            model = combo.model()
            for index in range(combo.count()):
                if combo.itemText(index) not in labels:
                    continue
                item = getattr(model, "item", lambda _index: None)(index)
                if item is not None:
                    item.setEnabled(False)
                    item.setToolTip("Unavailable in this RE-call build")

        def _config_controls(self) -> tuple[Any, ...]:
            return (
                self.embedder_combo,
                self.reranker_combo,
                self.splade_check,
                self.judge_combo,
                self.reasoning_combo,
                self.model_edit,
                self.reranker_model_edit,
                self.arm_combo,
            )

        def _capture_config(self) -> dict[str, Any]:
            return {
                "embedder": self.embedder_combo.currentText(),
                "reranker": self.reranker_combo.currentText(),
                "splade": self.splade_check.isChecked(),
                "judge": self.judge_combo.currentText(),
                "reasoning": self.reasoning_combo.currentText(),
                "model": self.model_edit.text(),
                "reranker_model": self.reranker_model_edit.text(),
                "arm": self.arm_combo.currentText(),
            }

        def _load_config_type(self, source_type: str) -> None:
            values = self._pipeline_configs[source_type]
            controls = self._config_controls()
            for control in controls:
                control.blockSignals(True)
            # `setCurrentText` is a NO-OP on a non-editable combo when the text is not an item, so
            # a saved value that no longer exists would leave the widget on its first entry while
            # `_pipeline_configs` still held the stale string — and the next Save would write that
            # stale string straight back. Restoring only what the combo can actually show, and
            # writing the fallback back into the config, keeps memory and screen saying one thing.
            for combo, key in (
                (self.embedder_combo, "embedder"),
                (self.reranker_combo, "reranker"),
                (self.judge_combo, "judge"),
                (self.reasoning_combo, "reasoning"),
                (self.arm_combo, "arm"),
            ):
                wanted = str(values[key])
                if combo.findText(wanted) >= 0:
                    combo.setCurrentText(wanted)
                else:
                    values[key] = combo.currentText()
            # SPLADE is restored only when this machine can actually run it. The probe DISABLES the
            # checkbox without unticking it, and `_save_configuration` refuses while it is ticked,
            # so a config saved on a CUDA machine (or one that hit a transient probe failure) would
            # restore a ticked, disabled box the user cannot clear: Save wedged permanently.
            values["splade"] = bool(values["splade"]) and self.splade_check.isEnabled()
            self.splade_check.setChecked(bool(values["splade"]))
            self.model_edit.setText(str(values["model"]))
            self.reranker_model_edit.setText(str(values["reranker_model"]))
            for control in controls:
                control.blockSignals(False)

        def _switch_config_type(self, source_type: str) -> None:
            if source_type == self._active_config_type:
                return
            self._pipeline_configs[self._active_config_type] = self._capture_config()
            self._active_config_type = source_type
            self._load_config_type(source_type)
            self.configuration_warning.setText(
                f"{source_type} has its own pipeline set. BGE and Voyage are unavailable. Any change requires a complete calibration."
            )

        def _build_calibration_page(self) -> QWidget:
            page = QWidget()
            layout = QVBoxLayout(page)
            title = QLabel("CORPUS CALIBRATION")
            title.setObjectName("pageTitle")
            layout.addWidget(title)
            safety = QLabel(
                "Select one row at a time. Run calibration test creates a draft only. It does not activate a new threshold. Review the draft before publishing it."
            )
            safety.setObjectName("warningLabel")
            safety.setWordWrap(True)
            layout.addWidget(safety)
            self.calibration_table = QTableWidget(0, 6)
            self.calibration_table.setObjectName("calibrationTable")
            self.calibration_table.setHorizontalHeaderLabels(
                ["PROJECT", "CORPUS", "STATUS", "LAST CALIBRATED", "CORPUS FINGERPRINT", "ACTIONS"]
            )
            self.calibration_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
            self.calibration_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
            self.calibration_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
            self.calibration_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
            self.calibration_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
            self.calibration_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Fixed)
            self.calibration_table.setColumnWidth(5, 190)
            self.calibration_table.verticalHeader().setVisible(False)
            self.calibration_table.verticalHeader().setDefaultSectionSize(64)
            self.calibration_table.setShowGrid(False)
            self.calibration_table.setItemDelegate(
                _FocuslessItemDelegate(self.calibration_table, hide_selection=True)
            )
            self.calibration_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            self.calibration_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.calibration_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
            self.calibration_table.itemSelectionChanged.connect(self._sync_calibration_action_surfaces)
            table_frame = QWidget()
            table_stack = QGridLayout(table_frame)
            table_stack.setContentsMargins(0, 0, 0, 0)
            table_stack.setSpacing(0)
            self._add_table_watermark(self.calibration_table, table_stack)
            table_stack.addWidget(self.calibration_table, 0, 0)

            refresh_actions = QFrame()
            refresh_actions.setObjectName("queueActions")
            refresh_layout = QHBoxLayout(refresh_actions)
            refresh_layout.setContentsMargins(6, 5, 6, 5)
            refresh_layout.setSpacing(6)
            self.refresh_calibration_button = QPushButton("Refresh")
            self.refresh_calibration_button.clicked.connect(self._refresh_calibration_table)
            refresh_layout.addWidget(self.refresh_calibration_button)
            self.cancel_calibration_button = QPushButton("Cancel")
            self.cancel_calibration_button.clicked.connect(self._cancel_selected_calibrations)
            refresh_layout.addWidget(self.cancel_calibration_button)
            table_stack.addWidget(
                refresh_actions,
                0,
                0,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
            )

            run_actions = QFrame()
            run_actions.setObjectName("queueActions")
            run_layout = QHBoxLayout(run_actions)
            run_layout.setContentsMargins(6, 5, 6, 5)
            run_layout.setSpacing(6)
            self.run_selected_calibration_button = QPushButton("Run")
            self.run_selected_calibration_button.clicked.connect(self._run_selected_calibration)
            run_layout.addWidget(self.run_selected_calibration_button)
            table_stack.addWidget(
                run_actions,
                0,
                0,
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
            )
            layout.addWidget(table_frame, 1)
            self.calibration_page_status = QLabel("")
            self.calibration_page_status.setObjectName("mutedLabel")
            layout.addWidget(self.calibration_page_status)
            return page

        def _build_settings_page(self) -> QWidget:
            page = QWidget()
            layout = QVBoxLayout(page)
            title = QLabel("SETTINGS")
            title.setObjectName("pageTitle")
            layout.addWidget(title)
            user_settings = QGroupBox("User settings")
            user_settings.setObjectName("watermarkGroup")
            user_form = QFormLayout(user_settings)
            self.language_combo = QComboBox()
            self.language_combo.addItems(["English", "Italiano", "Deutsch", "Español"])
            self.language_combo.currentTextChanged.connect(
                lambda language: self.settings_status.setText(f"Language preference set to {language}.")
            )
            user_form.addRow("Language", self.language_combo)
            layout.addWidget(user_settings)

            provider_keys = QGroupBox("Provider API keys")
            provider_keys.setObjectName("watermarkGroup")
            provider_layout = QVBoxLayout(provider_keys)
            provider_layout.setSpacing(10)
            key_form = QFormLayout()
            key_form.setVerticalSpacing(8)
            self.openrouter_key_edit = QLineEdit()
            self.openrouter_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
            self.openrouter_key_edit.setPlaceholderText("OpenRouter key")
            self.voyage_key_edit = QLineEdit()
            self.voyage_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
            self.voyage_key_edit.setPlaceholderText("Voyage key")
            self.openai_key_edit = QLineEdit()
            self.openai_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
            self.openai_key_edit.setPlaceholderText("OpenAI key")
            key_form.addRow("OpenRouter", self.openrouter_key_edit)
            key_form.addRow("Voyage", self.voyage_key_edit)
            key_form.addRow("OpenAI", self.openai_key_edit)
            provider_layout.addLayout(key_form)
            key_actions = QHBoxLayout()
            key_actions.addStretch()
            save_keys = QPushButton("Save API keys")
            save_keys.clicked.connect(self._save_api_keys)
            key_actions.addWidget(save_keys)
            provider_layout.addLayout(key_actions)
            layout.addWidget(provider_keys)

            updates = QGroupBox("RE-call updates")
            updates.setObjectName("watermarkGroup")
            update_layout = QHBoxLayout(updates)
            update_main = QVBoxLayout()
            self.update_result_label = QLabel("No release check has been run from this page.")
            self.update_result_label.setObjectName("mutedLabel")
            self.update_result_label.setWordWrap(True)
            update_actions = QHBoxLayout()
            self.check_update_button = QPushButton("Check for updates")
            self.check_update_button.clicked.connect(self._check_update_from_settings)
            self.apply_update_button = QPushButton("Apply update")
            self.apply_update_button.setEnabled(False)
            self.apply_update_button.clicked.connect(self._apply_update_from_settings)
            update_actions.addWidget(self.check_update_button)
            update_actions.addWidget(self.apply_update_button)
            update_actions.addStretch()
            update_main.addLayout(update_actions)
            update_layout.addLayout(update_main, 2)
            update_info = QVBoxLayout()
            self.version_label = QLabel(f"Client version: {self._current_version()}")
            update_info.addWidget(self.version_label)
            self.update_result_label.setMaximumWidth(250)
            update_info.addWidget(self.update_result_label)
            update_info.addStretch()
            update_layout.addLayout(update_info, 1)
            layout.addWidget(updates)

            layout.addWidget(self._build_database_group())

            info = QGroupBox("Runtime information")
            info.setObjectName("watermarkGroup")
            info_layout = QHBoxLayout(info)
            info_form = QFormLayout()
            info_form.addRow("Endpoint / compose", QLabel(self.profile.endpoint or self.profile.compose_file or "Not configured"))
            info_form.addRow("Pinned RE-call version", QLabel(self.profile.pinned_version or "Not pinned"))
            info_layout.addLayout(info_form, 2)
            runtime_info = QVBoxLayout()
            runtime_caption = QLabel("Runtime")
            runtime_caption.setObjectName("mutedLabel")
            runtime_info.addWidget(runtime_caption)
            runtime_value = QLabel(self.profile.mode.value)
            runtime_value.setObjectName("successLabel")
            runtime_info.addWidget(runtime_value)
            self.settings_status: QLabel = QLabel("Settings are stored locally for this prototype.")
            self.settings_status.setObjectName("mutedLabel")
            self.settings_status.setWordWrap(True)
            self.settings_status.setMaximumWidth(250)
            runtime_info.addSpacing(8)
            runtime_info.addWidget(self.settings_status)
            runtime_info.addStretch()
            info_layout.addLayout(runtime_info, 1)
            layout.addWidget(info)
            layout.addStretch()
            self._add_page_watermark(page)
            return page

        #: Display text for each runtime, in the order the settings page offers them. The stored
        #: value is the enum; this is only what a person reads.
        _MODE_LABELS = {
            RuntimeMode.DOCKER: "Managed Docker stack (RE-call installs and runs PostgreSQL)",
            RuntimeMode.LOCAL_DATABASE: "A PostgreSQL I already run (no Docker)",
            RuntimeMode.VPS_MCP: "A remote RE-call server (MCP endpoint)",
        }

        def _build_database_group(self) -> QWidget:
            """Where the install options that already existed become selectable.

            The engine has always accepted a database somebody else runs — `HeadlessConfig` takes
            `dsn` as the alternative to `data_root`, and with it set no compose file is written at
            all. Nothing in this window could say so, so Docker was the only reachable choice.
            """
            box = QGroupBox("Database")
            box.setObjectName("watermarkGroup")
            outer = QVBoxLayout(box)
            form = QFormLayout()

            self.mode_combo = QComboBox()
            for mode, label in self._MODE_LABELS.items():
                self.mode_combo.addItem(label, mode)
            current = self.mode_combo.findData(self.profile.mode)
            if current >= 0:
                self.mode_combo.setCurrentIndex(current)
            self.mode_combo.currentIndexChanged.connect(self._database_mode_changed)
            form.addRow("Runtime", self.mode_combo)

            self.dsn_edit = QLineEdit(self.profile.dsn or "")
            self.dsn_edit.setPlaceholderText("postgresql://user:password@127.0.0.1:5432/recall")
            form.addRow("Connection", self.dsn_edit)
            outer.addLayout(form)

            actions = QHBoxLayout()
            self.test_database_button = QPushButton("Test connection")
            self.test_database_button.clicked.connect(self._test_database)
            self.save_database_button = QPushButton("Save")
            self.save_database_button.clicked.connect(self._save_database)
            actions.addWidget(self.test_database_button)
            actions.addWidget(self.save_database_button)
            actions.addStretch()
            outer.addLayout(actions)

            self.database_status = QLabel(
                "Not tested. `Test connection` checks the server, the pgvector extension, "
                "permission to create tables, and whether an existing index matches this embedder."
            )
            self.database_status.setObjectName("mutedLabel")
            self.database_status.setWordWrap(True)
            outer.addWidget(self.database_status)

            self._database_mode_changed()
            return box

        def _selected_mode(self) -> RuntimeMode:
            """The chosen runtime, as the ENUM rather than whatever Qt handed back.

            ⛔ **Qt stores a `StrEnum` as a plain `str`.** Measured: `addItem(label,
            RuntimeMode.DOCKER)` then `itemData(0)` returns `'docker'`, type `str`. So
            `currentData() is RuntimeMode.DOCKER` is permanently False while
            `currentData() == RuntimeMode.DOCKER` is True, and identity is the comparison this file
            uses everywhere else for enums.

            Three call sites read that as "not the local-database mode", so the connection field
            never enabled, the test button never enabled, and Save took the branch that stores no
            DSN. Every one of them looked right. Converting once here means the rest of the class
            can keep comparing with `is`, which is what the reader expects.
            """
            return RuntimeMode(self.mode_combo.currentData())

        def _database_mode_changed(self) -> None:
            """Only one runtime takes a connection string, so only that one offers the field."""
            mode = self._selected_mode()
            needs_dsn = mode is RuntimeMode.LOCAL_DATABASE
            self.dsn_edit.setEnabled(needs_dsn)
            self.test_database_button.setEnabled(needs_dsn)
            if not needs_dsn:
                self.database_status.setText(
                    f"{self._MODE_LABELS[mode]} does not use a connection string."
                )

        def _database_settings(self) -> tuple[RuntimeMode, str] | None:
            """The chosen mode and a non-empty DSN, or None after reporting why not."""
            mode = self._selected_mode()
            dsn = self.dsn_edit.text().strip()
            if mode is RuntimeMode.LOCAL_DATABASE and not dsn:
                self.database_status.setText("Enter a connection string first.")
                return None
            return mode, dsn

        def _test_database(self) -> None:
            settings = self._database_settings()
            if settings is None:
                return
            _mode, dsn = settings
            self.test_database_button.setEnabled(False)
            self.database_status.setText("Testing...")
            # ⚠️ On the pool, never inline. `probe_database` opens a network connection, and a
            # database that is merely unreachable takes its whole timeout to say so — on the GUI
            # thread that is the window freezing with no way to tell it apart from a crash.
            self._run(
                lambda: probe_database(dsn),
                self._database_tested,
                self._database_test_failed,
            )

        def _database_tested(self, report: Any) -> None:
            self.test_database_button.setEnabled(True)
            self.database_status.setText(report.render())

        def _database_test_failed(self, message: str) -> None:
            self.test_database_button.setEnabled(True)
            self.database_status.setText(f"Could not test the connection: {self._safe(message)}")

        def _save_database(self) -> None:
            settings = self._database_settings()
            if settings is None:
                return
            mode, dsn = settings
            self.save_database_button.setEnabled(False)
            if mode is not RuntimeMode.LOCAL_DATABASE:
                self._persist_profile(mode, None)
                return
            self.database_status.setText("Checking the database before saving...")
            # Tested before it is written, deliberately. A profile that names a database nothing can
            # serve from is the silent-nothing failure with an extra step: the app restarts, the
            # runtime fails, and the setting that caused it looks like the one the user chose.
            self._run(
                lambda: probe_database(dsn),
                lambda report: self._save_checked(report, mode, dsn),
                self._database_save_failed,
            )

        def _save_checked(self, report: Any, mode: RuntimeMode, dsn: str) -> None:
            if not report.usable:
                self.save_database_button.setEnabled(True)
                self.database_status.setText(
                    "Not saved, because this database cannot serve RE-call yet.\n" + report.render()
                )
                return
            self._persist_profile(mode, dsn)

        def _persist_profile(self, mode: RuntimeMode, dsn: str | None) -> None:
            try:
                updated = replace(self.profile, mode=mode, dsn=dsn)
                save_profile(updated)
            except (OSError, ValueError) as exc:
                self.save_database_button.setEnabled(True)
                self.database_status.setText(f"Could not save: {self._safe(str(exc))}")
                return
            self.profile = updated
            self.save_database_button.setEnabled(True)
            # ⚠️ Says RESTART, because this window built its runtime at startup and does not rebuild
            # it. Reporting "saved" alone would leave the user watching the old runtime and
            # concluding the setting did nothing.
            self.database_status.setText(
                f"Saved. {self._MODE_LABELS[mode]}.\nRestart RE-call for it to take effect."
            )

        def _database_save_failed(self, message: str) -> None:
            self.save_database_button.setEnabled(True)
            self.database_status.setText(f"Could not save: {self._safe(message)}")

        def _safe(self, message: str) -> str:
            """Anything derived from a connection attempt, with this DSN's password removed.

            ⚠️ **These two paths carry an EXCEPTION, not a report.** `probe_database` scrubs what it
            returns, so the ordinary failures arrive clean; these handlers fire when the probe
            itself raised, and `_Worker` hands on a bare `str(exc)`. The label is on screen, in
            screenshots, and in whatever a user pastes into an issue.

            The comment that used to sit here simply asserted "Redacted" and nothing did any
            redacting. That is worse than no comment, because it stops the next reader looking.
            """
            return scrub_dsn_secrets(message, self.dsn_edit.text().strip())

        def _save_api_keys(self) -> None:
            self._api_keys = {
                "OpenRouter": self.openrouter_key_edit.text().strip(),
                "Voyage": self.voyage_key_edit.text().strip(),
                "OpenAI": self.openai_key_edit.text().strip(),
            }
            self.settings_status.setText(
                "API keys saved for this session. They will be used by the configured provider when the installer secure store is connected."
            )

        def _configuration_changed(self, *_args: Any) -> None:
            self._config_dirty = True
            self._calibration_required = True
            self.configuration_warning.setText(
                "Configuration changed. Save it, then run a complete calibration before activating the new set."
            )

        def _save_configuration(self) -> None:
            if self.splade_check.isChecked() and not self.splade_check.isEnabled():
                self.status.setText("SPLADE cannot be enabled because this machine does not meet the CUDA and VRAM requirements.")
                return
            forbidden = ("bge", "voyage")
            selected_models = (self.model_edit.text().lower(), self.reranker_model_edit.text().lower())
            if any(token in model for model in selected_models for token in forbidden):
                self.status.setText("BGE and Voyage models are not available in this RE-call configuration.")
                return
            self._pipeline_configs[self._active_config_type] = self._capture_config()
            # ⚠️ Actually WRITE it. Until this line the button set an in-memory dict and then told
            # the user "Configuration saved", so reopening the app restored the defaults and the
            # status line was a false claim. Demonstrated before fixing: set the embedder model,
            # save, close, reopen, and the field read `hashing-64` again.
            try:
                save_pipelines(self._pipeline_configs)
            except OSError as exc:
                # Say so rather than claim success. The choices are still live in this session, so
                # the user can keep working; what they cannot do is rely on them surviving.
                self.status.setText(
                    f"Configuration applied for this session but NOT saved: {exc.strerror or exc}"
                )
                self._config_dirty = False
                self._calibration_required = True
                return
            self._config_dirty = False
            self._calibration_required = True
            self.configuration_warning.setText(
                f"{self._active_config_type} configuration saved. A complete calibration is required before this set can be activated."
            )
            self.status.setText("Configuration saved. Open Calibration to certify the new set.")

        def _probe_hardware(self) -> Any:
            from recall.wizard.probe import probe_system

            return probe_system()

        def _hardware_probe_done(self, result: Any) -> None:
            from recall.wizard.probe import splade_is_feasible

            hardware = getattr(result, "hardware", None)
            cuda = bool(getattr(hardware, "cuda_available", False))
            vram = getattr(result, "cuda_vram_bytes", None)
            feasible = splade_is_feasible(cuda_available=cuda, vram_bytes=vram)
            self.splade_check.setEnabled(feasible)
            if feasible:
                gib = vram / (1024**3) if isinstance(vram, int) else 0
                self.splade_status.setText(f"CUDA available, {gib:.1f} GiB VRAM detected")
                self.splade_status.setObjectName("successLabel")
            else:
                self.splade_status.setText("Unavailable: CUDA GPU with at least 6 GiB VRAM is required")
            self.splade_status.style().unpolish(self.splade_status)
            self.splade_status.style().polish(self.splade_status)

        def _hardware_probe_failed(self, message: str) -> None:
            self.splade_check.setEnabled(False)
            self.splade_status.setText(f"Hardware check failed: {message}")

        def _calibration_targets(self) -> list[tuple[str, str, str]]:
            """One row per corpus that exists, naming the tenant it actually reports on.

            ⚠️ **The Memory row used to read the DOCS tenant.** `("Memory", "docs")` meant two rows
            showed the same corpus under different names, so the Memory row reported a calibration
            belonging to something else — and it read as reassuring, because the docs corpus is the
            one that certifies. The memory corpus is deliberately never calibrated, so its honest
            status is "missing"; showing another corpus's certification in its place is worse than
            showing nothing.

            "All projects" appears only when the runtime can serve that scope, for the same reason
            the scope selector hides it: the wizard provisions no `user-*` tenant.
            """
            projects = [(name, name) for name in self._project_names]
            if self._can_serve_shared():
                projects.append(("All projects", self.profile.shared_profile))
            targets: list[tuple[str, str, str]] = []
            for label, project in projects:
                for corpus, suffix in (
                    ("Documents", "docs"),
                    ("Code", "code"),
                    ("Memory", "memory"),
                ):
                    targets.append((label, corpus, f"{project}-{suffix}"))
            return targets

        def _refresh_calibration_table(self) -> None:
            if self._calibration_running:
                self.calibration_page_status.setText("A calibration is already running. Wait for it to finish before refreshing.")
                return
            targets = self._calibration_targets()
            self._calibration_targets_by_row = targets
            self.calibration_table.setRowCount(len(targets))
            for row, (project, corpus, physical_tenant) in enumerate(targets):
                for column, value in enumerate((project, corpus, "Checking…", "", "")):
                    self.calibration_table.setItem(row, column, QTableWidgetItem(value))
                self._set_calibration_actions(row, physical_tenant, None)
            if targets:
                self.calibration_table.selectRow(0)
            self.calibration_page_status.setText("Reading calibration artifacts and corpus metadata…")
            self._run(self._fetch_calibrations, self._calibrations_loaded, self._calibration_failed)

        def _fetch_calibrations(self) -> list[tuple[int, str, Any]]:
            cache: dict[str, Any] = {}
            results: list[tuple[int, str, Any]] = []
            for row, (_project, _corpus, physical_tenant) in enumerate(self._calibration_targets_by_row):
                if physical_tenant not in cache:
                    cache[physical_tenant] = self.runtime.calibration_status(physical_tenant)
                results.append((row, physical_tenant, cache[physical_tenant]))
            return results

        def _calibrations_loaded(self, results: list[tuple[int, str, Any]]) -> None:
            for row, physical_tenant, snapshot in results:
                self._calibration_results[physical_tenant] = snapshot
                self._populate_calibration_row(row, physical_tenant, snapshot)
            self.calibration_page_status.setText("")
            self._set_calibration_controls_enabled(True)

        def _calibration_failed(self, message: str) -> None:
            self._calibration_running = False
            self._set_calibration_controls_enabled(True)
            self.calibration_page_status.setText(f"Calibration status unavailable: {message}")

        def _populate_calibration_row(self, row: int, physical_tenant: str, snapshot: Any) -> None:
            raw = getattr(snapshot, "raw", {}) or {}
            status = str(getattr(snapshot, "status", "unknown"))
            if self._needs_calibration(snapshot):
                status = "RECALIBRATION SUGGESTED"
            elif status == "certified":
                status = "CERTIFIED"
            last = raw.get("published_at") or raw.get("calibrated_at") or raw.get("created_at") or "Not calibrated"
            fingerprint = raw.get("corpus_fingerprint") or "Not available"
            self.calibration_table.setItem(row, 2, QTableWidgetItem(status))
            self.calibration_table.setItem(row, 3, QTableWidgetItem(str(last)))
            self.calibration_table.setItem(row, 4, QTableWidgetItem(str(fingerprint)))
            self._set_calibration_actions(row, physical_tenant, snapshot)

        def _needs_calibration(self, snapshot: Any) -> bool:
            status = str(getattr(snapshot, "status", "unknown")).lower()
            raw = getattr(snapshot, "raw", {}) or {}
            if self._calibration_required or self._config_dirty:
                return True
            if raw.get("corpus_changed") is True or raw.get("stale") is True:
                return True
            active_generation = raw.get("active_generation_id") or raw.get("current_generation_id")
            if active_generation and getattr(snapshot, "generation_id", None) and active_generation != snapshot.generation_id:
                return True
            active_fingerprint = raw.get("active_corpus_fingerprint") or raw.get("current_corpus_fingerprint")
            if active_fingerprint and raw.get("corpus_fingerprint") and active_fingerprint != raw.get("corpus_fingerprint"):
                return True
            return status in {"missing", "stale", "rejected", "superseded"}

        def _set_calibration_actions(self, row: int, physical_tenant: str, snapshot: Any) -> None:
            cell = QWidget()
            cell.setObjectName("calibrationActionsCell")
            actions = QHBoxLayout(cell)
            actions.setContentsMargins(8, 10, 8, 10)
            actions.setSpacing(8)
            cell.setMinimumWidth(190)
            run = QPushButton("Run")
            run.setObjectName("tableActionButton")
            run.setFixedSize(62, 40)
            run.clicked.connect(lambda _checked=False, target_row=row: self._run_calibration_row(target_row))
            actions.addWidget(run)
            publish = QPushButton("Publish")
            publish.setObjectName("tableActionButton")
            publish.setFixedSize(86, 40)
            publish.setEnabled(bool(getattr(snapshot, "calibration_id", None)))
            publish.clicked.connect(
                lambda _checked=False, target_row=row: self._publish_calibration_row(target_row)
            )
            actions.addWidget(publish)
            self.calibration_table.setCellWidget(row, 5, cell)
            self._sync_calibration_action_surfaces()

        def _sync_calibration_action_surfaces(self) -> None:
            for row in range(self.calibration_table.rowCount()):
                cell = self.calibration_table.cellWidget(row, 5)
                if cell is None:
                    continue
                cell.setStyleSheet("background: transparent;")

        def _run_calibration_row(self, row: int) -> None:
            if row >= len(self._calibration_targets_by_row):
                return
            if self._calibration_running:
                self.calibration_page_status.setText("Another calibration is already running. Wait for it to finish.")
                return
            project, corpus, physical_tenant = self._calibration_targets_by_row[row]
            confirmation = QMessageBox.question(
                self,
                "Start calibration test?",
                f"Run a calibration test for {project} / {corpus}?\n\nThis creates a draft artifact and uses runtime resources. It will not activate anything until you review and publish the draft.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if confirmation != QMessageBox.StandardButton.Yes:
                self.calibration_page_status.setText("Calibration test cancelled. No artifact was changed.")
                return
            self._calibration_running = True
            self._set_calibration_controls_enabled(False)
            self.calibration_page_status.setText(f"Running real calibration for {corpus} in {physical_tenant}…")
            self._run(
                lambda: self.runtime.run_calibration(physical_tenant),
                lambda snapshot: self._calibration_row_done(row, physical_tenant, snapshot),
                self._calibration_failed,
            )

        def _calibration_row_done(self, row: int, physical_tenant: str, snapshot: Any) -> None:
            self._calibration_running = False
            self._calibration_results[physical_tenant] = snapshot
            self.calibration_snapshot = snapshot
            self._populate_calibration_row(row, physical_tenant, snapshot)
            self._set_calibration_controls_enabled(True)
            self.calibration_page_status.setText(
                f"Calibration draft created for {physical_tenant}. Publish it after reviewing the result."
            )

        def _publish_calibration_row(self, row: int) -> None:
            if row >= len(self._calibration_targets_by_row):
                return
            _project, corpus, physical_tenant = self._calibration_targets_by_row[row]
            snapshot = self._calibration_results.get(physical_tenant)
            calibration_id = getattr(snapshot, "calibration_id", None)
            if not calibration_id:
                self.calibration_page_status.setText(f"Run calibration for {corpus} before publishing.")
                return
            if self._calibration_running:
                self.calibration_page_status.setText("Another calibration is already running. Wait for it to finish.")
                return
            confirmation = QMessageBox.question(
                self,
                "Publish calibration draft?",
                f"Publish the calibration draft for {corpus} in {physical_tenant}?\n\nPublishing activates this calibration for retrieval. Confirm only after reviewing the draft results.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if confirmation != QMessageBox.StandardButton.Yes:
                self.calibration_page_status.setText("Publish cancelled. The draft remains unchanged.")
                return
            self._calibration_running = True
            self._set_calibration_controls_enabled(False)
            self._run(
                lambda: self.runtime.publish_calibration(physical_tenant, calibration_id),
                lambda result: self._calibration_row_done(row, physical_tenant, result),
                self._calibration_failed,
            )

        def _run_selected_calibration(self) -> None:
            row = self.calibration_table.currentRow()
            if row < 0:
                self.calibration_page_status.setText("Select one corpus row before starting calibration.")
                return
            self._run_calibration_row(row)

        def _cancel_selected_calibrations(self) -> None:
            rows = sorted(
                {index.row() for index in self.calibration_table.selectionModel().selectedRows()}
            )
            if not rows:
                self.calibration_page_status.setText("Select one or more corpus rows before cancelling.")
                return
            if self._calibration_running:
                self.calibration_page_status.setText("A calibration is already running. Wait for it to finish before cancelling.")
                return
            count = len(rows)
            confirmation = QMessageBox.question(
                self,
                "Confirm",
                f"Cancel the selected calibration action{'s' if count != 1 else ''}?\n\nNo corpus data or published calibration artifact will be deleted.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if confirmation != QMessageBox.StandardButton.Yes:
                self.calibration_page_status.setText("Cancellation cancelled. The calibration selection is unchanged.")
                return
            self.calibration_table.clearSelection()
            self.calibration_table.setCurrentCell(-1, -1)
            self.calibration_page_status.setText(
                f"Cancelled {count} selected calibration action{'s' if count != 1 else ''}. No corpus data was changed."
            )

        def _set_calibration_controls_enabled(self, enabled: bool) -> None:
            self.calibration_table.setEnabled(enabled)

        def _current_version(self) -> str:
            try:
                return version("recall-rag")
            except PackageNotFoundError:
                return self.profile.pinned_version or "0.0.0"

        def _check_update_from_settings(self) -> None:
            self.check_update_button.setEnabled(False)
            self.update_result_label.setText("Checking signed release metadata…")
            self._run(self._check_update_safely, self._settings_update_done, self._settings_update_failed)

        def _settings_update_done(self, release: Any) -> None:
            self.check_update_button.setEnabled(True)
            self._update_checked(release)
            if release is None:
                self.update_result_label.setText("No update information is available.")
                return
            self.update_result_label.setText(
                f"Latest release: {release.version} ({release.asset_name or 'installer asset'})."
            )

        def _settings_update_failed(self, message: str) -> None:
            self.check_update_button.setEnabled(True)
            self.update_result_label.setText(f"Update check failed: {message}")

        def _apply_update_from_settings(self) -> None:
            release = self._latest_release
            if release is None:
                return
            self.apply_update_button.setEnabled(False)
            self.update_result_label.setText(f"Applying RE-call {release.version}…")
            self._run(
                lambda: self.runtime.apply_update(release),
                self._update_applied,
                self._update_apply_failed,
            )

        def _update_applied(self, result: Any) -> None:
            self.update_result_label.setText("Update applied and runtime health checked successfully.")
            self.status.setText("RE-call update applied successfully.")

        def _update_apply_failed(self, message: str) -> None:
            self.apply_update_button.setEnabled(True)
            self.update_result_label.setText(f"Update was not applied: {message}")

        def _scope_data(self) -> dict[str, Any]:
            data = self.scope.currentData()
            if isinstance(data, dict):
                return data
            return {"tenant": self.profile.default_tenant, "shared": bool(data)}

        def _can_serve_shared(self) -> bool:
            """Whether the shared "all projects" scope actually exists in this stack.

            Asked of the runtime rather than assumed, because the answer differs by install: the
            legacy compose defines `recall-user-docs`/`recall-user-code`, and a wizard-generated
            stack defines neither. Offering a scope nothing can serve is the same defect class as a
            status line claiming a save that did not happen — the menu asserts a capability.

            Any runtime that cannot answer the question keeps the entry. A VPS server resolves the
            scope remotely and this process cannot know what it holds, so hiding on ignorance would
            remove a working choice; only a definite "no service for this scope" hides it.
            """
            resolve = getattr(self.runtime, "_service_for_tenant", None)
            if resolve is None:
                return True
            for kind in ("docs", "code", "memory"):
                try:
                    resolve(f"{self.profile.shared_profile}-{kind}")
                except Exception:  # noqa: BLE001 - any refusal means "not servable", not a crash
                    continue
                return True
            return False

        def _populate_scopes(self, tenants: list[str]) -> None:
            names: list[str] = []
            for raw in tenants:
                name = _project_name(str(raw))
                if name and name != self.profile.shared_profile and name not in names:
                    names.append(name)
            if self.profile.default_tenant not in names:
                names.insert(0, self.profile.default_tenant)
            self._project_names = names
            if hasattr(self, "github_tenant_combo"):
                self._populate_github_tenants(names)
            self.scope.blockSignals(True)
            self.scope.clear()
            for name in names:
                self.scope.addItem(name, {"tenant": name, "shared": False})
            # ⚠️ Offered only when the runtime can actually serve it. The wizard provisions
            # `<project>-*` and never a `user-*` scope, so on every wizard install this entry
            # resolved to a tenant with no MCP service: a permanent menu item that refused whenever
            # anyone picked it. The legacy `docker-compose.desktop.yml` DOES define
            # `recall-user-docs` and `recall-user-code`, which is why it looked fine there.
            if self._can_serve_shared():
                self.scope.addItem(
                    "All projects (shared memory)",
                    {"tenant": self.profile.shared_profile, "shared": True},
                )
            self.scope.addItem("+ Add project", {"action": "add"})
            preferred = self.scope.findText(self.profile.default_tenant)
            self.scope.setCurrentIndex(preferred if preferred >= 0 else 0)
            self._last_scope_index = self.scope.currentIndex()
            self.scope.blockSignals(False)

        def _provision_project(self, name: str) -> None:
            """Actually create the project, rather than only offering its name.

            "+ Add project" used to append a string to this combo box and stop. The scope then
            pointed at a tenant with no compose service, no MCP server block and no corpus, so it
            did not survive a restart and anything reaching the runtime with it was refused — a
            failure that arrived later, worded as a problem with a "tenant scope".

            The corpora are NOT built or calibrated here. That takes minutes and the user has asked
            for somewhere to put files, not for an index of files they have not chosen yet; the
            queue page fills it and the calibration page certifies it. So the project arrives real
            but empty and uncalibrated, and this says so instead of implying it is ready.

            Docker only. `RuntimeManager._call_for` discards the tenant argument, so in VPS mode the
            scope is a field sent to a remote server and provisioning it is that server's business,
            not something this process can do or should claim to have done.
            """
            if not isinstance(self.runtime, DockerRuntime):
                self.status.setText(
                    f"Project {name!r} is selected. This app provisions projects only for the "
                    f"managed local stack; on a remote server the scope has to exist there."
                )
                return

            compose_file = self.profile.compose_file
            if not compose_file:
                self.status.setText(
                    f"Cannot create {name!r}: this profile names no compose file, so there is no "
                    f"stack to add it to."
                )
                return

            # ⚠️ **On the worker pool, not the GUI thread.** This ran `add_project` and a full
            # `runtime.start()` inline in a `currentIndexChanged` slot, so the window stopped
            # repainting and Windows marked it "(Not Responding)" for the length of a compose
            # `up --wait` — which, on a first start, includes building the image. Every other
            # long operation in this class already goes through `_run`; this was the only one that
            # did not. Four auditors found it.
            self.status.setText(f"Creating project {name!r}…")
            self.scope.setEnabled(False)
            self._run(
                lambda: self._do_provision(name, compose_file),
                lambda outcome: self._provision_done(name, outcome),
                lambda message: self._provision_failed(name, message),
            )

        def _do_provision(self, name: str, compose_file: str) -> tuple[object, str]:
            """Worker half: create the project and bring the stack back up. No Qt here.

            Returns `(added, start_error)`. ⚠️ **Creating and starting are separate outcomes, and
            collapsing them lost a state the user has to be told about.** Once `add_project`
            returns, the compose file HAS been written and the tenants exist; if the restart then
            fails, "cannot create" is false and dropping the name from the selector strands the
            user — a retry finds the tenants already present, reports "already exists", and never
            starts anything.
            """
            # Hand over the compose path the profile ACTUALLY records, not just its directory.
            # Passing the parent alone discarded the filename and let `add_project` guess it,
            # and the guess was wrong: the installer writes `docker-compose.recall.yml`.
            from recall.wizard.projects import add_project

            stack_file = Path(compose_file).resolve()
            added = add_project(stack_file.parent, name, compose_path=stack_file)
            try:
                # Unconditionally, not only when something was created: a retry after a failed
                # start finds nothing to add, and skipping the start there is what left the user
                # with no route back to a running stack. `start()` is idempotent.
                self.runtime.start()
            except Exception as exc:  # noqa: BLE001 - reported, not handled; see the docstring
                return added, str(exc)
            return added, ""

        def _provision_done(self, name: str, outcome: tuple[object, str]) -> None:
            self.scope.setEnabled(True)
            added, start_error = outcome
            created = getattr(added, "tenants", ())

            if start_error:
                # The project EXISTS — the compose file was written before the start was attempted.
                # Saying "cannot create" here would be a claim about a state the code did create,
                # and the name stays in the selector so a retry can start it.
                where = getattr(added, "compose_path", None)
                self.status.setText(
                    f"Project {name!r} was written to {where.name if where else 'the stack'}, but "
                    f"the stack did not come back up: {start_error}. The project exists; it will "
                    f"be there on the next connect."
                )
                return

            if not created:
                self.status.setText(f"Project {name!r} already exists in this stack; selected it.")
                return

            # Names the pages as the navigation labels them. This said "the Queue page", which the
            # user cannot find: the buttons are MAIN, GITHUB, CALIBRATION, CONFIG, SETTINGS, and
            # `queue_page` is only the internal name of the one shown as MAIN. (Finding DOC-005.)
            self.status.setText(
                f"Created project {name!r} ({len(created)} corpora). It is empty and not yet "
                f"calibrated: add files on the MAIN page, then calibrate it on the CALIBRATION "
                f"page before trusting search."
            )

        def _provision_failed(self, name: str, message: str) -> None:
            self.scope.setEnabled(True)
            # Every failure shape lands here, including the ones `_Worker` converts from arbitrary
            # exceptions. The previous inline version caught only `RuntimeErrorBase`, while
            # `runtime.start()` reaches the MCP client and can raise types that are not.
            self.status.setText(f"Cannot create {name!r}: {message}")
            self._forget_project(name)

        def _forget_project(self, name: str) -> None:
            """Drop a name that was added to the combo box but could not be provisioned."""
            if name in self._project_names:
                self._project_names.remove(name)
            self._populate_scopes(self._project_names)

        def _scope_changed(self, index: int) -> None:
            data = self.scope.itemData(index)
            if isinstance(data, dict) and data.get("action") == "add":
                name, accepted = QInputDialog.getText(self, "Add project", "Project name")
                clean_name = _project_name(name) if accepted else ""
                if clean_name:
                    provisioned = clean_name in self._project_names
                    if not provisioned:
                        self._project_names.append(clean_name)
                        self._populate_scopes(self._project_names)
                    selected = self.scope.findText(clean_name)
                    if selected >= 0:
                        self.scope.setCurrentIndex(selected)
                    else:
                        # The name was REFUSED by `_populate_scopes`, which drops anything equal to
                        # the shared profile and then resets the index to the default project. So
                        # the user is now on a scope they did not choose. Saying "is selected" here
                        # would be an affirmative false claim about the live write target, and the
                        # next drop would ingest into the default corpus.
                        self.scope.blockSignals(True)
                        self.scope.setCurrentIndex(self._last_scope_index)
                        self.scope.blockSignals(False)
                        self.status.setText(
                            f"{clean_name!r} is reserved for the shared scope, which is already "
                            f"listed as 'All projects (shared memory)'. Selection unchanged."
                        )
                        return
                    if not provisioned:
                        self._provision_project(clean_name)
                else:
                    self.scope.blockSignals(True)
                    self.scope.setCurrentIndex(self._last_scope_index)
                    self.scope.blockSignals(False)
                return
            self._last_scope_index = index
            self._refresh_table()

        def _populate_github_tenants(self, tenants: list[str]) -> None:
            if not hasattr(self, "github_tenant_combo"):
                return
            current = self.github_tenant_combo.currentData()
            self.github_tenant_combo.blockSignals(True)
            self.github_tenant_combo.clear()
            for name in tenants:
                self.github_tenant_combo.addItem(name, {"tenant": name, "shared": False})
            self.github_tenant_combo.addItem(
                "All projects (shared memory)",
                {"tenant": self.profile.shared_profile, "shared": True},
            )
            preferred = 0
            if isinstance(current, dict):
                for index in range(self.github_tenant_combo.count()):
                    if self.github_tenant_combo.itemData(index) == current:
                        preferred = index
                        break
            self.github_tenant_combo.setCurrentIndex(preferred)
            self.github_tenant_combo.blockSignals(False)

        def _github_scope_category(self) -> SourceCategory | None:
            choice = self.github_scope_combo.currentText()
            if choice == "Code only":
                return SourceCategory.CODE
            if choice == "Documents only":
                return SourceCategory.DOCUMENTS
            return None

        def _github_scope_data(self) -> dict[str, Any]:
            data = self.github_tenant_combo.currentData()
            if isinstance(data, dict):
                return dict(data)
            return {"tenant": self.profile.default_tenant, "shared": False}

        def _download_github(self) -> None:
            url = self.github_url_edit.text().strip()
            if not url:
                self.github_status.setText("Enter a GitHub repository URL first.")
                self.github_url_edit.setFocus()
                return
            self.github_download_button.setEnabled(False)
            self.github_approve_button.setEnabled(False)
            self.github_download_button.setText("Downloading…")
            self.github_status.setText("Downloading repository and filtering supported files…")
            category = self._github_scope_category()
            self._run(
                lambda: download_repository(url, category),
                self._github_downloaded,
                self._github_download_failed,
            )

        def _github_downloaded(self, result: GithubImport) -> None:
            self.github_download_button.setEnabled(True)
            self.github_download_button.setText("DOWNLOAD")
            self.github_import = result
            self.github_root = result.root
            self.github_pending = list(result.files)
            self._refresh_github_table()
            self.github_approve_button.setEnabled(bool(self.github_pending))
            self.github_status.setText(
                f"Downloaded {len(self.github_pending)} supported file(s) from "
                f"{result.owner}/{result.repository}. Review the list, then approve it."
            )

        def _github_download_failed(self, message: str) -> None:
            self.github_download_button.setEnabled(True)
            self.github_download_button.setText("DOWNLOAD")
            self.github_approve_button.setEnabled(False)
            self.github_status.setText(message)
            QMessageBox.warning(self, "GitHub download", message)

        def _refresh_github_table(self) -> None:
            tenant = self._scope_label(self._github_scope_data())
            self.github_table.setRowCount(0)
            for path, category in self.github_pending:
                row = self.github_table.rowCount()
                self.github_table.insertRow(row)
                relative = path.name
                if self.github_root is not None:
                    try:
                        relative = str(path.relative_to(self.github_root))
                    except ValueError:
                        relative = path.name
                self.github_table.setItem(row, 0, QTableWidgetItem(relative))
                self.github_table.setItem(row, 1, QTableWidgetItem(tenant))
                self.github_table.setItem(row, 2, QTableWidgetItem(display_type(path, category)))

        def _clear_github(self) -> None:
            self.github_import = None
            self.github_root = None
            self.github_pending.clear()
            self.github_table.setRowCount(0)
            self.github_approve_button.setEnabled(False)
            self.github_status.setText("Enter a repository URL to begin.")

        def _approve_github(self) -> None:
            if not self.github_pending:
                self.github_status.setText("Download a repository before approving files.")
                return
            scope = self._github_scope_data()
            count = len(self.github_pending)
            self.pending_files.extend(self.github_pending)
            self.pending_scopes.extend(dict(scope) for _ in self.github_pending)
            self._refresh_table()
            self.start_button.setEnabled(True)
            self.status.setText(
                f"{count} GitHub file(s) approved. Review the Main queue, then start indexing."
            )
            self._show_page(0)

        def _accept_drop(self, raw_paths: list[str]) -> None:
            detected: list[tuple[Path, SourceCategory]] = []
            for path in collect_files([Path(raw_path) for raw_path in raw_paths], None):
                category = classify(path) or SourceCategory.MEMORY
                detected.append((path, category))
            if not detected:
                self.status.setText(
                    "No supported files were found. Use UTF-8 text or code files (.md, .txt, .py, .js, and similar)."
                )
                return
            default_scope = dict(self._scope_data())
            self.pending_files.extend(detected)
            self.pending_scopes.extend(dict(default_scope) for _ in detected)
            self._refresh_table()
            self.start_button.setEnabled(True)
            self.status.setText(f"{len(detected)} file(s) added. Confirm the tenant, then start indexing.")

        def _browse(self) -> None:
            chosen, _ = QFileDialog.getOpenFileNames(self, "Choose source files")
            if chosen:
                self._accept_drop(chosen)

        def _scan_requested(self) -> None:
            confirmation = QMessageBox.question(
                self,
                "Scan local sources",
                "Search Documents, Desktop, Downloads, and Claude memory folders for supported files?\n\n"
                "Files are added to the queue for review. Nothing is indexed until you press Start indexing.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if confirmation != QMessageBox.StandardButton.Yes:
                self.status.setText("Local scan cancelled. The queue is unchanged.")
                return
            self.scan_button.setEnabled(False)
            self.scan_button.setText("Scanning…")
            self.status.setText("Scanning the selected local folders…")
            self._run(self._scan_local_sources, self._scan_completed, self._scan_failed)

        def _scan_local_sources(self) -> list[tuple[Path, SourceCategory]]:
            queued = {
                str(path.expanduser().resolve()).casefold()
                for path, _category in self.pending_files
            }
            detected: list[tuple[Path, SourceCategory]] = []
            for path in scan_files(default_scan_roots()):
                key = str(path.expanduser().resolve()).casefold()
                if key in queued:
                    continue
                filename = path.name.casefold()
                category = (
                    SourceCategory.MEMORY
                    if filename in CLAUDE_MEMORY_FILENAMES
                    or any(part.casefold() in {"memory", "memories"} for part in path.parts[:-1])
                    else classify(path)
                )
                if category is not None:
                    detected.append((path, category))
                    queued.add(key)
            return detected

        def _scan_completed(self, detected: list[tuple[Path, SourceCategory]]) -> None:
            self.scan_button.setEnabled(True)
            self.scan_button.setText("Scan")
            if not detected:
                self.status.setText(
                    "No new supported files found in the selected local folders."
                )
                return
            default_scope = dict(self._scope_data())
            self.pending_files.extend(detected)
            self.pending_scopes.extend(dict(default_scope) for _ in detected)
            self._refresh_table()
            self.start_button.setEnabled(True)
            self.status.setText(
                f"Scan added {len(detected)} new file(s) to the queue. "
                "Review the project, then start indexing. RE-call checks source fingerprints during indexing."
            )

        def _scan_failed(self, message: str) -> None:
            self.scan_button.setEnabled(True)
            self.scan_button.setText("Scan")
            self._job_failed(message)

        def _refresh_table(self) -> None:
            header = self.files.horizontalHeader()
            sorting = self.files.isSortingEnabled()
            sort_column = header.sortIndicatorSection()
            sort_order = header.sortIndicatorOrder()
            self.files.setSortingEnabled(False)
            self.files.setRowCount(0)
            for pending_index, (path, category) in enumerate(self.pending_files):
                scope = self._pending_scope(pending_index)
                tenant = self._scope_label(scope)
                row = self.files.rowCount()
                self.files.insertRow(row)
                file_item = QTableWidgetItem(path.name)
                file_item.setData(Qt.ItemDataRole.UserRole, pending_index)
                self.files.setItem(row, 0, file_item)
                self.files.setItem(row, 1, _HiddenSortItem(tenant))
                self.files.setItem(row, 2, QTableWidgetItem(display_type(path, category)))
                tenant_combo = QComboBox()
                tenant_combo.setObjectName("tenantCellCombo")
                tenant_combo.addItems(self._tenant_options())
                tenant_combo.blockSignals(True)
                tenant_combo.setCurrentText(tenant)
                tenant_combo.blockSignals(False)
                tenant_combo.setFixedHeight(42)
                tenant_combo.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
                tenant_combo.currentTextChanged.connect(
                    lambda value, index=pending_index: self._tenant_changed(index, value)
                )
                self.files.setCellWidget(row, 1, tenant_combo)
            self.files.setSortingEnabled(sorting)
            if sorting and sort_column >= 0:
                self.files.sortItems(sort_column, sort_order)

        def _update_file_sort_indicator(self, section: int, order: Qt.SortOrder) -> None:
            header = self.files.horizontalHeader()
            header.blockSignals(True)
            header.setSortIndicatorShown(order == Qt.SortOrder.AscendingOrder)
            if isinstance(header, _FileHeader):
                header.setDescendingSection(
                    section if order == Qt.SortOrder.DescendingOrder else -1
                )
            for column, label in enumerate(("FILE NAME", "TENANT", "TYPE")):
                item = self.files.horizontalHeaderItem(column)
                if item is None:
                    continue
                item.setText(label)
            header.blockSignals(False)

        def _tenant_options(self) -> list[str]:
            return [*self._project_names, "All projects (shared memory)"]

        def _scope_label(self, scope: dict[str, Any]) -> str:
            if bool(scope.get("shared")):
                return "All projects (shared memory)"
            return str(scope.get("tenant") or self.profile.default_tenant)

        def _scope_from_label(self, label: str) -> dict[str, Any]:
            shared = label == "All projects (shared memory)"
            return {
                "tenant": self.profile.shared_profile if shared else label,
                "shared": shared,
            }

        def _pending_scope(self, pending_index: int) -> dict[str, Any]:
            if pending_index < len(self.pending_scopes):
                return self.pending_scopes[pending_index]
            return dict(self._scope_data())

        def _pending_index_for_row(self, row: int) -> int | None:
            item = self.files.item(row, 0)
            if item is None:
                return None
            value = item.data(Qt.ItemDataRole.UserRole)
            return value if isinstance(value, int) else None

        def _selected_pending_indices(self) -> list[int]:
            indices = {
                pending_index
                for selected in self.files.selectionModel().selectedRows()
                if (pending_index := self._pending_index_for_row(selected.row())) is not None
            }
            return sorted(indices)

        def _select_pending_indices(self, pending_indices: set[int]) -> None:
            selection = self.files.selectionModel()
            selection.clearSelection()
            for row in range(self.files.rowCount()):
                pending_index = self._pending_index_for_row(row)
                if pending_index not in pending_indices:
                    continue
                model_index = self.files.model().index(row, 0)
                selection.select(
                    model_index,
                    QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
                )

        def _tenant_changed(self, pending_index: int, label: str) -> None:
            if not 0 <= pending_index < len(self.pending_files):
                return
            selected_indices = set(self._selected_pending_indices())
            targets = selected_indices if pending_index in selected_indices else {pending_index}
            scope = self._scope_from_label(label)
            for target in targets:
                while len(self.pending_scopes) <= target:
                    self.pending_scopes.append(dict(self._scope_data()))
                self.pending_scopes[target] = dict(scope)
            self._refresh_table()
            self._select_pending_indices(targets)
            self.status.setText(
                f"Tenant changed for {len(targets)} file{'s' if len(targets) != 1 else ''}."
            )

        def _clear_files(self) -> None:
            self.pending_files.clear()
            self.pending_scopes.clear()
            self.files.setRowCount(0)
            self.start_button.setEnabled(False)
            self.progress.setValue(0)
            self.status.setText("Drop files to create an indexing queue.")

        def _cancel_selected_files(self) -> None:
            pending_indices = self._selected_pending_indices()
            if not pending_indices:
                self.status.setText("Select one or more queued files before cancelling.")
                return
            count = len(pending_indices)
            confirmation = QMessageBox.question(
                self,
                "Confirm",
                f"Cancel and remove {count} selected file{'s' if count != 1 else ''} from the queue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if confirmation != QMessageBox.StandardButton.Yes:
                self.status.setText("Cancellation cancelled. The queue is unchanged.")
                return
            for pending_index in reversed(pending_indices):
                if 0 <= pending_index < len(self.pending_files):
                    del self.pending_files[pending_index]
                    if pending_index < len(self.pending_scopes):
                        del self.pending_scopes[pending_index]
            self._refresh_table()
            self.start_button.setEnabled(bool(self.pending_files))
            if not self.pending_files:
                self.progress.setValue(0)
            self.status.setText(f"Removed {count} file{'s' if count != 1 else ''} from the queue.")

        def _prepare_runtime(self) -> list[str]:
            self.runtime.start()
            return self.runtime.list_tenants()

        def _reconnect(self) -> None:
            self.reconnect_button.setEnabled(False)
            self.reconnect_button.setText("Searching…")
            self.status.setText("Searching for the RE-call runtime…")
            self._run(self._prepare_runtime, self._runtime_ready, self._runtime_failed)

        def _start_embedding(self) -> None:
            groups: dict[tuple[SourceCategory, str, bool], list[Path]] = {}
            for pending_index, (path, category) in enumerate(self.pending_files):
                scope = self._pending_scope(pending_index)
                tenant = str(scope.get("tenant") or self.profile.default_tenant)
                shared = bool(scope.get("shared"))
                groups.setdefault((category, tenant, shared), []).append(path)
            self.start_button.setEnabled(False)
            self.start_button.setText("Indexing…")
            self.progress.setValue(10)
            self.status.setText("Indexing is running…")
            self._run(
                lambda: [
                    self.runtime.start_ingest(
                        SourceSelection(
                            category,
                            tuple(paths),
                            tenant,
                            shared,
                            shared_profile=self.profile.shared_profile,
                        )
                    )
                    for (category, tenant, shared), paths in groups.items()
                ],
                self._job_done,
            )

        def _check_calibration(self) -> None:
            self._run(lambda: self.runtime.calibration_status(self._selected_tenant()), self._show_calibration)

        def _run_calibration(self) -> None:
            self.status.setText("Preparing calibration draft…")
            self._run(lambda: self.runtime.run_calibration(self._selected_tenant()), self._show_calibration)

        def _publish_calibration(self) -> None:
            snapshot = self.calibration_snapshot
            if snapshot is None or not snapshot.calibration_id:
                self.status.setText("Run or check calibration before publishing.")
                return
            self._run(
                lambda: self.runtime.publish_calibration(self._selected_tenant(), snapshot.calibration_id),
                self._show_calibration,
            )

        def _selected_tenant(self) -> str:
            scope = self._scope_data()
            if bool(scope.get("shared")):
                return f"{self.profile.shared_profile}-docs"
            tenant = str(scope.get("tenant") or self.profile.default_tenant)
            return tenant if tenant.endswith("-docs") else f"{tenant}-docs"

        def _show_calibration(self, result: Any) -> None:
            self.calibration_snapshot = result
            raw = getattr(result, "raw", {})
            generation = raw.get("generation_id") or result.generation_id or "none"
            corpus = raw.get("corpus_fingerprint") or "none"
            self.status.setText(
                f"Calibration: {result.status} | generation {generation} | corpus {corpus} | {result.message}"
            )

        def _run(self, fn: Any, done: Any, failed: Any | None = None) -> None:
            """Run `fn` on the pool and deliver its result to `done` (or `failed`).

            ⚠️ **The worker must be KEPT, or the callback is delivered by luck.** `_Worker` is a
            `QRunnable` with `autoDelete` on, so the moment `run()` returns Qt destroys it — taking
            `_WorkerSignals` with it and purging the queued cross-thread call before it is
            delivered. Measured on PySide6 6.11 with five identical jobs: **1 of 5 arrived** as this
            was written, **5 of 5** with a reference held. Zero in another run, because it is a
            garbage-collection race rather than a deterministic failure, which is worse: it works
            often enough to look correct.

            This affects every caller — connect, ingest, calibration, the GitHub download — not
            just the one that exposed it. It surfaced because provisioning disables the scope
            selector and re-enables it only from these callbacks, so a lost signal left the control
            disabled for good.
            """
            worker = _Worker(fn)
            worker.setAutoDelete(False)
            self._workers.append(worker)

            def _release(_: Any = None) -> None:
                # Idempotent: both signals are connected to it, and only one ever fires, but a
                # double release must not raise inside a Qt slot.
                if worker in self._workers:
                    self._workers.remove(worker)

            worker.signals.done.connect(done)
            worker.signals.failed.connect(failed or (lambda message: self._job_failed(message)))
            worker.signals.done.connect(_release)
            worker.signals.failed.connect(_release)
            self.pool.start(worker)

        def _runtime_ready(self, result: Any) -> None:
            tenants = result if isinstance(result, list) else []
            self._populate_scopes(tenants or [self.profile.default_tenant])
            if self.pending_files:
                self._refresh_table()
            self._set_runtime_state(True)
            self.reconnect_button.setEnabled(True)
            self.reconnect_button.setText("Reconnect")
            self.reconnect_button.setVisible(False)
            self.runtime_label.setText(f"Runtime: {self.profile.mode.value} ready")
            self.status.setText("Runtime ready. Drop a source to begin.")
            self._run(self._check_update_safely, self._update_checked)

        def _runtime_failed(self, message: str) -> None:
            self._set_runtime_state(False)
            self.reconnect_button.setEnabled(True)
            self.reconnect_button.setText("Reconnect")
            self.reconnect_button.setVisible(True)
            self.status.setText(f"Runtime unavailable: {message}")

        def _set_runtime_state(self, connected: bool) -> None:
            color = "#39d98a" if connected else "#ef6262"
            self.connection_light.setStyleSheet(f"background: {color}; border-radius: 5px;")
            state = "ready" if connected else "disconnected"
            self.runtime_label.setText(f"Runtime: {self.profile.mode.value} {state}")

        def _open_config(self) -> None:
            self._show_page(3)

        def _check_update_safely(self) -> Any:
            try:
                return self.runtime.check_update()
            except Exception:  # noqa: BLE001
                return None

        def _update_checked(self, release: Any) -> None:
            self._latest_release = release
            if hasattr(self, "apply_update_button"):
                self.apply_update_button.setEnabled(False)
            if release is None:
                return
            from recall.desktop.updates import is_newer

            current = self._current_version()
            newer = is_newer(current, release.version)
            if hasattr(self, "apply_update_button"):
                self.apply_update_button.setEnabled(newer)
            if newer:
                self.status.setText(
                    f"RE-call update available: {release.version}. Apply it from Settings."
                )
            elif hasattr(self, "update_result_label"):
                self.update_result_label.setText(f"RE-call {current} is up to date.")

        def _job_done(self, result: Any) -> None:
            self.start_button.setEnabled(True)
            self.start_button.setText("Start indexing")
            self.progress.setValue(100)
            if isinstance(result, list):
                self.status.setText(f"Indexed {len(result)} source group(s).")
            else:
                self.status.setText(getattr(result, "message", "Indexing completed."))

        def _job_failed(self, message: str) -> None:
            self.start_button.setEnabled(True)
            self.start_button.setText("Start indexing")
            self.status.setText(message)
            QMessageBox.warning(self, "RE-call", message)

        def closeEvent(self, event: Any) -> None:
            """Close, always, and never wait longer than a person will.

            ⛔ **`waitForDone()` with no argument waits FOREVER.** A provisioning worker sits inside
            `docker compose up`, whose timeout is `_SLOW_VERB_TIMEOUT` — 1800 seconds. So closing
            the window during a first install froze the whole application for up to half an hour,
            unresponsive, with no indication that anything was happening. On Windows that is the
            state the OS offers to kill for you, and killing it mid-provision is how a stack ends up
            half-created.

            It also made `test_provisioning_is_dispatched_to_the_pool_not_the_gui_thread` flaky
            rather than failing: the wait only exceeds the test timeout when Docker happens to be
            busy, so it passed on a quiet machine and hung on a loaded one. Found while running the
            suite against a database for the first time, which is not where the defect was.

            **Bounded, and closing regardless.** The work is in subprocesses with their own
            timeouts; abandoning the wait does not abandon the install, and Docker finishes what it
            started. The workers are retained in `self._workers` with `setAutoDelete(False)`, so a
            still-running one cannot be destroyed underneath its own signals — the same retention
            that fixed the garbage-collection race documented in `_run`.
            """
            try:
                finished = self.pool.waitForDone(_CLOSE_WAIT_MS)
                if not finished:
                    # Reported rather than silent. Something is still running and the user is
                    # entitled to know that closing the window did not stop it.
                    print(
                        f"recall: closing while {self.pool.activeThreadCount()} background task(s) "
                        "are still running; they continue in Docker and will finish on their own."
                    )
                self.runtime.stop()
            finally:
                event.accept()


def run_window(window: Any) -> int:
    """Show `window` and run the application until it closes.

    ⛔ **One launcher, because there are now two windows.** The graphical installer
    (`recall/desktop/install_ui.py`) needs exactly this and nothing else, and a second copy of it
    would be a second place that decides whether to reuse an existing `QApplication` — which is the
    line that decides whether the installer can be opened from inside the running desktop app or
    crashes with "A QApplication instance already exists".
    """
    if QApplication is None:
        raise RuntimeErrorBase('The desktop extra is required. Install with: pip install "recall-rag[desktop]"')
    app = QApplication.instance() or QApplication([])
    window.show()
    return int(app.exec())


def run_app(profile: RuntimeProfile) -> int:
    """The main desktop window. Unchanged behaviour; the launcher underneath it is now shared."""
    if QApplication is None:
        raise RuntimeErrorBase('The desktop extra is required. Install with: pip install "recall-rag[desktop]"')
    return run_window(MainWindow(profile))
