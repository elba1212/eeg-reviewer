"""Native Windows review UI. Heavy analysis runs in a cancellable QProcess."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
import uuid

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from .recordings import Recording, canonical_electrode
from .reporting import POSITIONS, STAGE_Y, clock, pct, report_html
from .scoring import summarize
from .storage import Store, write_json

Qt = QtCore.Qt


class FrameModel(QtCore.QAbstractTableModel):
    edited = QtCore.Signal()
    validation_error = QtCore.Signal(str)

    def __init__(self, frame, columns, editable=(), max_time=None):
        super().__init__()
        self.frame = frame
        self.columns = [c for c in columns if c in frame]
        self.editable = set(editable)
        self.max_time = max_time

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self.frame)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        value = self.frame.iloc[index.row()][self.columns[index.column()]]
        if role == Qt.ItemDataRole.UserRole:
            return index.row()
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            if pd.isna(value) or value == "":
                return ""
            if isinstance(value, (float, np.floating)):
                return f"{value:.3f}" if role == Qt.ItemDataRole.DisplayRole else str(value)
            return str(value)
        if role == Qt.ItemDataRole.BackgroundRole and self.columns[index.column()] == "status":
            return QtGui.QColor({"accepted":"#e0f2e8", "rejected":"#f6e4e5", "uncertain":"#fff0d2"}.get(str(value),"#edf1f5"))
        if role == Qt.ItemDataRole.ToolTipRole and self.columns[index.column()] == "confidence":
            return "Model confidence is not a guarantee of stage accuracy."
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            return self.columns[section].replace("_", " ").title() if orientation == Qt.Orientation.Horizontal else str(section + 1)

    def flags(self, index):
        flags = super().flags(index)
        if index.isValid() and self.columns[index.column()] in self.editable:
            flags |= Qt.ItemFlag.ItemIsEditable
        return flags

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if role != Qt.ItemDataRole.EditRole or self.columns[index.column()] not in self.editable:
            return False
        column = self.columns[index.column()]
        allowed = {"stage": ["W","N1","N2","N3","R","UNKNOWN"], "status": ["unreviewed","accepted","rejected","uncertain"]}
        if column in allowed and value not in allowed[column]:
            return False
        if column in ("start_sec","end_sec"):
            try:
                value = float(value)
                row = self.frame.iloc[index.row()]
                start = value if column == "start_sec" else float(row.start_sec)
                end = value if column == "end_sec" else float(row.end_sec)
                if not np.isfinite([start,end]).all() or start < 0 or end <= start or (self.max_time is not None and end > self.max_time):
                    raise ValueError
            except (TypeError,ValueError):
                self.validation_error.emit("Event times must be finite, inside the recording, and end after start.")
                return False
        self.frame.iat[index.row(), self.frame.columns.get_loc(column)] = value
        self.dataChanged.emit(index,index,[role])
        self.edited.emit()
        return True


class ChoiceDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, choices, parent=None):
        super().__init__(parent)
        self.choices = choices

    def createEditor(self, parent, option, index):
        editor = QtWidgets.QComboBox(parent)
        editor.addItems(self.choices)
        return editor

    def setEditorData(self, editor, index):
        editor.setCurrentText(str(index.data(Qt.ItemDataRole.EditRole)))

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentText())


def button(text, slot, primary=False):
    widget = QtWidgets.QPushButton(text)
    widget.clicked.connect(slot)
    if primary:
        widget.setProperty("primary", True)
    return widget


def table_view(model):
    view = QtWidgets.QTableView()
    view.setModel(model)
    view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
    view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
    view.setAlternatingRowColors(True)
    view.verticalHeader().hide()
    view.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
    view.horizontalHeader().setStretchLastSection(True)
    view.setSortingEnabled(False)
    view.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked | QtWidgets.QAbstractItemView.EditTrigger.SelectedClicked)
    view.setWordWrap(False)
    return view


class IntervalDialog(QtWidgets.QDialog):
    def __init__(self, study, start, end, title, channel=False, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        layout = QtWidgets.QFormLayout(self)
        self.start = QtWidgets.QDoubleSpinBox()
        self.end = QtWidgets.QDoubleSpinBox()
        for widget in [self.start, self.end]:
            widget.setRange(0, study.duration)
            widget.setDecimals(3)
            widget.setSuffix(" s")
        self.start.setValue(start)
        self.end.setValue(min(end,study.duration))
        self.reason = QtWidgets.QLineEdit()
        self.reason.setPlaceholderText("Optional note")
        self.channel = QtWidgets.QComboBox()
        self.channel.addItems(["All"] + study.picks)
        layout.addRow("Start from recording", self.start)
        layout.addRow("End from recording", self.end)
        if channel:
            layout.addRow("Channel", self.channel)
        layout.addRow("Note", self.reason)
        actions = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        actions.accepted.connect(self.validate)
        actions.rejected.connect(self.reject)
        layout.addRow(actions)

    def validate(self):
        if self.end.value() <= self.start.value():
            QtWidgets.QMessageBox.warning(self,"Interval","End must be after start.")
            return
        self.accept()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, workspace):
        super().__init__()
        self.store = Store(workspace)
        self.study = None
        self.frames = None
        self.recording = None
        self.process = None
        self.models = {}
        self.position = 0.0
        self.stdout_buffer = ""
        self.setWindowTitle("EEG Reviewer — Sleep & Epileptic Activity")
        self.resize(1440,940)
        self.setMinimumSize(1050,720)
        pg.setConfigOptions(background="w", foreground="#294259", antialias=True)
        body = QtWidgets.QWidget()
        self.setCentralWidget(body)
        layout = QtWidgets.QHBoxLayout(body)
        sidebar = QtWidgets.QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(255)
        rail = QtWidgets.QVBoxLayout(sidebar)
        brand = QtWidgets.QLabel("EEG REVIEWER")
        brand.setObjectName("brand")
        rail.addWidget(brand)
        rail.addWidget(QtWidgets.QLabel("Sleep & epileptic activity"))
        rail.addSpacing(20)
        self.import_button = button("+ Import EEG", self.import_eeg, True)
        rail.addWidget(self.import_button)
        rail.addWidget(QtWidgets.QLabel("RECORDING LIBRARY"))
        self.library = QtWidgets.QListWidget()
        self.library.currentItemChanged.connect(self.open_selected)
        rail.addWidget(self.library,1)
        self.workspace_button = button("Change workspace…", self.change_workspace)
        rail.addWidget(self.workspace_button)
        rail.addWidget(button("Import existing project analyses",lambda: self.start_job("discover")))
        privacy = QtWidgets.QLabel("LOCAL WORKSPACE\nRecordings and reports stay\non this computer.")
        privacy.setObjectName("privacy")
        rail.addWidget(privacy)
        layout.addWidget(sidebar)
        main = QtWidgets.QVBoxLayout()
        layout.addLayout(main,1)
        header = QtWidgets.QHBoxLayout()
        self.title = QtWidgets.QLabel("Choose a recording")
        self.title.setObjectName("title")
        header.addWidget(self.title,1)
        self.analyze_button = button("Run analysis", self.analyze, True)
        self.export_button = button("Export report", self.export)
        header.addWidget(self.analyze_button)
        header.addWidget(self.export_button)
        main.addLayout(header)
        self.subtitle = QtWidgets.QLabel("Import a hospital EEG or open a prepared study.")
        self.subtitle.setWordWrap(True)
        main.addWidget(self.subtitle)
        self.notice = QtWidgets.QLabel("PRELIMINARY PERSONAL REVIEW · Automatic stages and event flags require expert review.")
        self.notice.setObjectName("notice")
        self.notice.setWordWrap(True)
        main.addWidget(self.notice)
        self.tabs = QtWidgets.QTabWidget()
        main.addWidget(self.tabs,1)
        self.build_overview()
        self.build_eeg()
        self.build_channels()
        self.build_review()
        self.build_report()
        footer = QtWidgets.QHBoxLayout()
        self.status = QtWidgets.QLabel("Ready")
        self.status.setWordWrap(True)
        footer.addWidget(self.status,1)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setFixedWidth(180)
        self.progress.hide()
        footer.addWidget(self.progress)
        self.cancel_button = button("Cancel",self.cancel_job)
        self.cancel_button.hide()
        footer.addWidget(self.cancel_button)
        main.addLayout(footer)
        self.notes_timer = QtCore.QTimer(self)
        self.notes_timer.setSingleShot(True)
        self.notes_timer.timeout.connect(self.save_notes)
        self.job_poll_timer = QtCore.QTimer(self)
        self.job_poll_timer.setInterval(500)
        self.job_poll_timer.timeout.connect(self.read_job_progress)
        self.refresh_library()

    def build_overview(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        cards = QtWidgets.QHBoxLayout()
        self.cards = {}
        for key, text in [("sleep","Scored sleep"),("nrem","NREM sleep"),("swi","N2/N3 SWI estimate"),("reviewed","Reviewed SWI")]:
            frame = QtWidgets.QFrame()
            frame.setObjectName("card")
            column = QtWidgets.QVBoxLayout(frame)
            column.addWidget(QtWidgets.QLabel(text))
            value = QtWidgets.QLabel("—")
            value.setObjectName("metric")
            column.addWidget(value)
            self.cards[key] = value
            cards.addWidget(frame)
        layout.addLayout(cards)
        self.coverage_label = QtWidgets.QLabel()
        self.coverage_label.setWordWrap(True)
        layout.addWidget(self.coverage_label)
        self.hypno_plot = pg.PlotWidget(title="Hypnogram · click to inspect EEG")
        self.hypno_plot.setMinimumHeight(180)
        self.hypno_plot.getAxis("left").setTicks([[(0,"N3"),(1,"N2"),(2,"N1"),(3,"REM"),(4,"Wake")]])
        self.hypno_plot.setLabel("bottom","Minutes from recording start")
        self.hypno_plot.scene().sigMouseClicked.connect(lambda event: self.plot_clicked(event,self.hypno_plot))
        layout.addWidget(self.hypno_plot,1)
        self.swi_plot = pg.PlotWidget(title="N2/N3 SWI estimate · 30-minute intervals")
        self.swi_plot.setYRange(0,100)
        self.swi_plot.setLabel("left","SWI","%")
        self.swi_plot.setLabel("bottom","Minutes from recording start")
        self.swi_plot.setXLink(self.hypno_plot)
        self.swi_plot.scene().sigMouseClicked.connect(lambda event: self.plot_clicked(event,self.swi_plot))
        layout.addWidget(self.swi_plot,1)
        self.seizure_label = QtWidgets.QLabel("Seizure screening has not been run.")
        self.seizure_label.setWordWrap(True)
        self.seizure_label.setObjectName("notice")
        layout.addWidget(self.seizure_label)
        self.tabs.addTab(page,"Overview")

    def build_eeg(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        navigation = QtWidgets.QHBoxLayout()
        navigation.addWidget(button("◀ Previous",lambda: self.move_eeg(-self.window_seconds.currentData())))
        navigation.addWidget(button("Next ▶",lambda: self.move_eeg(self.window_seconds.currentData())))
        self.time_input = QtWidgets.QDoubleSpinBox()
        self.time_input.setRange(0,1_000_000)
        self.time_input.setDecimals(1)
        self.time_input.setSuffix(" s from start")
        self.time_input.editingFinished.connect(lambda: self.show_eeg(self.time_input.value()))
        navigation.addWidget(self.time_input)
        self.window_seconds = QtWidgets.QComboBox()
        for seconds in [10,30,60,120]:
            self.window_seconds.addItem(f"{seconds} seconds",seconds)
        self.window_seconds.setCurrentIndex(1)
        self.window_seconds.currentIndexChanged.connect(lambda: self.show_eeg(self.position))
        navigation.addWidget(self.window_seconds)
        self.spacing = QtWidgets.QComboBox()
        for value in [100,200,500,1000]:
            self.spacing.addItem(f"{value} µV spacing",value)
        self.spacing.setCurrentIndex(1)
        self.spacing.currentIndexChanged.connect(lambda: self.show_eeg(self.position))
        navigation.addWidget(self.spacing)
        navigation.addStretch()
        layout.addLayout(navigation)
        self.eeg_plot = pg.PlotWidget()
        self.eeg_plot.setLabel("bottom","Seconds from recording start")
        self.eeg_plot.showGrid(x=True,y=False,alpha=.15)
        layout.addWidget(self.eeg_plot,2)
        self.eeg_hint = QtWidgets.QLabel("Unfiltered source traces · spike candidates: amber · possible seizures: red · excluded intervals: gray")
        self.eeg_hint.setWordWrap(True)
        layout.addWidget(self.eeg_hint)
        actions = QtWidgets.QHBoxLayout()
        actions.addWidget(button("Add spike interval",lambda: self.add_event("spikes")))
        actions.addWidget(button("Mark possible seizure",lambda: self.add_event("seizures")))
        actions.addWidget(button("Exclude artifact interval",lambda: self.add_interval("artifact")))
        actions.addWidget(button("Mark interval reviewed",lambda: self.add_interval("review")))
        layout.addLayout(actions)
        self.tabs.addTab(page,"EEG review")

    def build_channels(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        hint = QtWidgets.QLabel("Verify electrode labels and reference before analysis. Original channel names stay unchanged. Check Detection for SWI channels; seizure screening checks its own required montage.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        settings = QtWidgets.QFormLayout()
        self.study_name = QtWidgets.QLineEdit()
        self.reference = QtWidgets.QLineEdit()
        self.age = QtWidgets.QDoubleSpinBox()
        self.age.setRange(-1,120)
        self.age.setSpecialValueText("Unknown")
        self.age.setSuffix(" years")
        self.eeg_channel = QtWidgets.QComboBox()
        self.eog_channel = QtWidgets.QComboBox()
        self.emg_channel = QtWidgets.QComboBox()
        for name, widget in [("Study name",self.study_name),("Age at recording",self.age),("Recorded reference",self.reference),("Staging EEG",self.eeg_channel),("Optional EOG",self.eog_channel),("Optional EMG",self.emg_channel)]:
            settings.addRow(name,widget)
        layout.addLayout(settings)
        self.channels = QtWidgets.QTableWidget(0,5)
        self.channels.setHorizontalHeaderLabels(["Source channel","Electrode / derivation","Role","Exclude","SWI detection"])
        self.channels.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.channels,1)
        layout.addWidget(button("Save electrode settings", self.save_channels, True))
        self.tabs.addTab(page,"Electrodes & setup")

    def build_review(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        self.review_tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.review_tabs)
        self.event_views = {}
        for key, label in [("hypnogram","Sleep stages"),("spikes","Spike candidates"),("seizures","Possible seizures")]:
            inner = QtWidgets.QWidget()
            column = QtWidgets.QVBoxLayout(inner)
            if key != "hypnogram":
                toolbar = QtWidgets.QHBoxLayout()
                for status in ["accepted","rejected","uncertain","unreviewed"]:
                    toolbar.addWidget(button(status.title(),lambda checked=False,k=key,s=status:self.set_event_status(k,s)))
                toolbar.addStretch()
                column.addLayout(toolbar)
            placeholder = FrameModel(pd.DataFrame(),[])
            view = table_view(placeholder)
            view.doubleClicked.connect(lambda index,k=key:self.jump_to_row(k,index.row()))
            column.addWidget(view)
            self.event_views[key] = view
            self.review_tabs.addTab(inner,label)
        interval_page = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(interval_page)
        column.addWidget(QtWidgets.QLabel("Reviewed SWI uses only explicitly reviewed N2/N3 seconds without unresolved spike candidates."))
        self.interval_view = table_view(FrameModel(pd.DataFrame(),[]))
        column.addWidget(self.interval_view)
        column.addWidget(button("Remove selected user intervals",self.remove_intervals))
        self.review_tabs.addTab(interval_page,"Artifacts & review coverage")
        distribution_page = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(distribution_page)
        text = QtWidgets.QLabel("Scalp field distribution depends on montage and reference. These electrodes do not establish an intracranial source or seizure-onset zone.")
        text.setWordWrap(True)
        column.addWidget(text)
        self.scalp_plot = pg.PlotWidget()
        self.scalp_plot.setAspectLocked()
        self.scalp_plot.hideAxis("left")
        self.scalp_plot.hideAxis("bottom")
        column.addWidget(self.scalp_plot,1)
        self.distribution_view = table_view(FrameModel(pd.DataFrame(),[]))
        column.addWidget(self.distribution_view,1)
        self.review_tabs.addTab(distribution_page,"Scalp distribution")
        layout.addWidget(QtWidgets.QLabel("Double-click a stage or event to inspect its EEG. Edit stage/status cells to record a review decision."))
        self.tabs.addTab(page,"Stages & events")

    def build_report(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.addWidget(QtWidgets.QLabel("Review notes — saved automatically and included in the draft report"))
        self.notes = QtWidgets.QPlainTextEdit()
        self.notes.setMaximumHeight(125)
        self.notes.setPlaceholderText("Add observations, questions for the neurologist, or contextual information…")
        self.notes.textChanged.connect(lambda: self.notes_timer.start(500) if hasattr(self,"notes_timer") else None)
        layout.addWidget(self.notes)
        layout.addWidget(button("Refresh draft preview", self.preview_report))
        self.report_browser = QtWidgets.QTextBrowser()
        self.report_browser.setOpenExternalLinks(False)
        layout.addWidget(self.report_browser,1)
        self.tabs.addTab(page,"Draft report")

    def refresh_library(self, selected_id=None):
        self.library.blockSignals(True)
        self.library.clear()
        selected_item = None
        for study in self.store.list_studies():
            item = QtWidgets.QListWidgetItem(study.name.replace("_"," "))
            item.setData(Qt.ItemDataRole.UserRole, study.id)
            item.setToolTip(f"{study.first_timestamp or 'Acquisition date unknown'}\n{clock(study.duration)} · {len(study.channels)} channels")
            self.library.addItem(item)
            if study.id == selected_id:
                selected_item = item
        self.library.blockSignals(False)
        if selected_item or self.library.count():
            self.library.setCurrentItem(selected_item or self.library.item(0))
        self.update_buttons()

    def open_selected(self,current,previous=None):
        if not current or self.process:
            return
        self.save_notes()
        try:
            if self.recording:
                self.recording.close()
            self.study = self.store.load(current.data(Qt.ItemDataRole.UserRole))
            self.frames = self.store.load_frames(self.study)
            self.recording = Recording(self.study)
            self.title.setText(self.study.name.replace("_"," "))
            self.subtitle.setText(f"{clock(self.study.duration)} · {self.study.sfreq:g} Hz · {len(self.study.channels)} channels · {self.study.first_timestamp or 'Acquisition time unknown'}")
            self.notes.blockSignals(True)
            self.notes.setPlainText(self.study.narrative)
            self.notes.blockSignals(False)
            self.populate_channels()
            self.populate_review()
            self.refresh_metrics()
            self.show_eeg(0)
            self.preview_report()
        except Exception as exc:
            self.error(str(exc))
        self.update_buttons()

    def populate_channels(self):
        study = self.study
        self.study_name.setText(study.name)
        self.reference.setText(study.reference)
        self.age.setValue(study.age if study.age is not None else -1)
        for combo, selection in [(self.eeg_channel,study.eeg_channel),(self.eog_channel,study.eog_channel),(self.emg_channel,study.emg_channel)]:
            combo.clear()
            combo.addItems([""] + study.channels)
            combo.setCurrentText(selection)
        self.channels.setRowCount(len(study.channels))
        for index, ch in enumerate(study.channels):
            source = QtWidgets.QTableWidgetItem(ch)
            source.setFlags(source.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.channels.setItem(index,0,source)
            self.channels.setItem(index,1,QtWidgets.QTableWidgetItem(study.mapping.get(ch,ch)))
            role = QtWidgets.QComboBox()
            role.addItems(["EEG","EOG","EMG","Other"])
            role.setCurrentText(study.roles.get(ch,"Other"))
            self.channels.setCellWidget(index,2,role)
            for col, checked in [(3,ch in study.excluded),(4,ch in study.detection_channels)]:
                item = QtWidgets.QTableWidgetItem()
                item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
                item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
                self.channels.setItem(index,col,item)

    def save_channels(self):
        if not self.study or self.process:
            return
        before = asdict(self.study)
        self.study.name = self.study_name.text().strip() or self.study.name
        self.study.reference = self.reference.text().strip() or "Unknown / as recorded"
        self.study.age = None if self.age.value() < 0 else self.age.value()
        self.study.eeg_channel = self.eeg_channel.currentText()
        self.study.eog_channel = self.eog_channel.currentText()
        self.study.emg_channel = self.emg_channel.currentText()
        self.study.excluded = []
        self.study.detection_channels = []
        for index,ch in enumerate(self.study.channels):
            self.study.mapping[ch] = self.channels.item(index,1).text().strip()
            self.study.roles[ch] = self.channels.cellWidget(index,2).currentText()
            if self.channels.item(index,3).checkState() == Qt.CheckState.Checked:
                self.study.excluded.append(ch)
            if self.channels.item(index,4).checkState() == Qt.CheckState.Checked:
                self.study.detection_channels.append(ch)
        changed = any(before[key] != asdict(self.study)[key] for key in ["mapping","roles","excluded","detection_channels","eeg_channel","eog_channel","emg_channel","age","reference"])
        self.study.settings_changed = self.study.settings_changed or (changed and bool(self.study.active_run))
        self.store.save(self.study)
        self.title.setText(self.study.name)
        self.status.setText("Settings saved. Run analysis again to update results." if self.study.settings_changed else "Settings saved.")
        self.refresh_metrics()
        self.update_buttons()

    def populate_review(self):
        columns = {"hypnogram":["start_sec","end_sec","original_stage","stage","confidence"],
                   "spikes":["start_sec","end_sec","channel","status","amplitude_uv","robust_z","origin"],
                   "seizures":["start_sec","end_sec","channel","status","score","origin"]}
        for key,view in self.event_views.items():
            field = "stage" if key == "hypnogram" else "status"
            model = FrameModel(self.frames[key],columns[key],[field] if key == "hypnogram" else [field,"start_sec","end_sec"],self.study.duration)
            self.models[key] = model
            model.edited.connect(self.review_changed)
            model.validation_error.connect(self.status.setText)
            view.setModel(model)
            delegate = ChoiceDelegate(["W","N1","N2","N3","R","UNKNOWN"] if key == "hypnogram" else ["unreviewed","accepted","rejected","uncertain"],view)
            view.setItemDelegateForColumn(model.columns.index(field),delegate)
            view.resizeColumnsToContents()
        self.refresh_intervals()

    def refresh_intervals(self):
        rows = [{**s,"kind":"artifact"} for s in self.study.bad_spans] + [{**s,"kind":"review"} for s in self.study.reviewed_spans]
        self.interval_frame = pd.DataFrame(rows,columns=["kind","start_sec","end_sec","reason","origin"])
        self.interval_view.setModel(FrameModel(self.interval_frame,list(self.interval_frame.columns)))
        self.interval_view.resizeColumnsToContents()

    def refresh_metrics(self):
        if not self.study or self.frames is None:
            return
        self.metrics = summarize(self.study,self.frames)
        m = self.metrics
        stale = self.study.settings_changed
        self.notice.setText("ELECTRODE SETTINGS CHANGED · Reanalysis is required before interpreting or exporting these results." if stale else "PRELIMINARY PERSONAL REVIEW · Automatic stages and event flags require expert review.")
        values = {"sleep":clock(m["sleep_seconds"]),"nrem":clock(m["nrem_sleep_seconds"]),"swi":pct(m["swi"]["percent"]),"reviewed":pct(m["reviewed_swi"]["percent"])}
        for key,value in values.items():
            self.cards[key].setText("Reanalysis needed" if stale else value)
        self.coverage_label.setText(f"Eligible N2/N3: {clock(m['swi']['eligible_seconds'])} · positive bins: {m['swi']['positive_seconds']} · reviewed N2/N3: {clock(m['reviewed_swi']['eligible_seconds'])} · unscored/excluded: {clock(m['unscored_seconds'])}")
        self.hypno_plot.clear()
        hyp = self.frames["hypnogram"]
        if len(hyp):
            x,y = [],[]
            for row in hyp.itertuples(index=False):
                x.extend([row.start_sec/60,min(row.end_sec,self.study.duration)/60,np.nan])
                y.extend([STAGE_Y.get(row.stage,np.nan)]*2+[np.nan])
            self.hypno_plot.plot(x,y,pen=pg.mkPen("#16877d",width=2),connect="finite")
            self.hypno_plot.setYRange(-.3,4.4)
            self.hypno_plot.setXRange(0,self.study.duration/60,padding=.01)
        self.swi_plot.clear()
        segments = [s for s in m["segments"] if s["percent"] is not None]
        if segments and not stale:
            self.swi_plot.addItem(pg.BarGraphItem(x0=[s["start_sec"]/60 for s in segments],x1=[s["end_sec"]/60 for s in segments],height=[s["percent"] for s in segments],brush="#16877d",pen="w"))
        analysis = self.frames["analysis"]
        status = analysis.get("seizure_status","not_run")
        count = len(m["seizures"])
        self.seizure_label.setText((f"{count} possible seizure event(s) remain for review. " if count else "No candidates detected in analyzed intervals. ") + analysis.get("seizure_message","") if status == "completed" else "Seizure screening unavailable / not run. " + analysis.get("seizure_message",""))
        self.draw_scalp(m["channels"])

    def draw_scalp(self,channels):
        self.scalp_plot.clear()
        theta = np.linspace(0,2*np.pi,160)
        self.scalp_plot.plot(np.cos(theta),np.sin(theta),pen="#64748b")
        self.scalp_plot.plot([-.12,0,.12],[1,1.12,1],pen="#64748b")
        for channel in channels:
            label = canonical_electrode(channel["electrode"])
            if label not in POSITIONS:
                continue
            x,y = POSITIONS[label]
            value = channel["percent"] or 0
            color = QtGui.QColor.fromHsvF(.13*(1-value/100),.25+.65*value/100,.95)
            self.scalp_plot.addItem(pg.ScatterPlotItem([x],[y],size=22,brush=color,pen="#516174"))
            text = pg.TextItem(f"{label}\n{pct(channel['percent'])}",anchor=(.5,0),color="#294259")
            text.setPos(x,y-.06)
            self.scalp_plot.addItem(text)
        self.scalp_plot.setRange(xRange=(-1.3,1.3),yRange=(-1.35,1.3))
        frame = pd.DataFrame(channels)
        self.distribution_view.setModel(FrameModel(frame,list(frame.columns)))
        self.distribution_view.resizeColumnsToContents()

    def show_eeg(self,start):
        if not self.recording or not self.study:
            return
        try:
            seconds = self.window_seconds.currentData()
            self.position = max(0,min(float(start),max(0,self.study.duration-seconds)))
            end = min(self.study.duration,self.position+seconds)
            self.time_input.blockSignals(True)
            self.time_input.setValue(self.position)
            self.time_input.blockSignals(False)
            picks = self.study.picks or [ch for ch in self.study.channels if ch not in self.study.excluded][:12]
            values = self.recording.get_uv(picks,self.position,end)
            times = self.position+np.arange(values.shape[1])/self.study.sfreq
            spacing = self.spacing.currentData()
            self.eeg_plot.clear()
            # Peak-preserving display downsampling is handled by PyQtGraph.
            for index,(ch,data) in enumerate(zip(picks,values)):
                curve = self.eeg_plot.plot(times,data+index*spacing,pen=pg.mkPen("#294259",width=.8),connect="finite")
                curve.setDownsampling(auto=True,method="peak")
                curve.setClipToView(True)
            self.eeg_plot.getAxis("left").setTicks([[(i*spacing,self.study.mapping.get(ch,ch)) for i,ch in enumerate(picks)]])
            self.eeg_plot.setXRange(self.position,end,padding=0)
            self.eeg_plot.setYRange(-spacing,len(picks)*spacing,padding=.01)
            for key,color in [("spikes",(239,177,68,55)),("seizures",(211,78,89,65))]:
                events = self.frames[key]
                if events.empty:
                    continue
                selected = events[(events.start_sec < end)&(events.end_sec > self.position)&(events.status != "rejected")]
                # Union overlapping event intervals keeps dense SWI recordings responsive.
                from .pipeline import merge_intervals
                for span in merge_intervals(selected.to_dict("records")):
                    self.eeg_plot.addItem(pg.LinearRegionItem([span["start_sec"],span["end_sec"]],movable=False,brush=color,pen=None))
            for span in self.study.bad_spans:
                if span["start_sec"] < end and span["end_sec"] > self.position:
                    self.eeg_plot.addItem(pg.LinearRegionItem([span["start_sec"],span["end_sec"]],movable=False,brush=(80,90,100,75),pen=None))
        except Exception as exc:
            self.error(f"Could not display EEG: {exc}")

    def move_eeg(self,amount):
        self.show_eeg(self.position+amount)

    def plot_clicked(self,event,plot):
        if self.study and event.button() == Qt.MouseButton.LeftButton and plot.sceneBoundingRect().contains(event.scenePos()):
            point = plot.getPlotItem().vb.mapSceneToView(event.scenePos())
            self.show_eeg(point.x()*60)
            self.tabs.setCurrentIndex(1)

    def jump_to_row(self,key,row):
        if self.frames is not None and 0 <= row < len(self.frames[key]):
            self.show_eeg(max(0,float(self.frames[key].iloc[row].start_sec)-3))
            self.tabs.setCurrentIndex(1)

    def set_event_status(self,key,status):
        if not self.study or not self.study.active_run or self.process:
            return
        selected = self.event_views[key].selectionModel().selectedRows()
        for index in selected:
            self.frames[key].loc[index.row(),"status"] = status
        if selected:
            self.models[key].layoutChanged.emit()
            self.review_changed()

    def review_changed(self):
        if not self.study or self.process:
            return
        try:
            self.store.save_reviews(self.study,self.frames)
            self.refresh_metrics()
            self.show_eeg(self.position)
            self.status.setText("Review saved. Scores and charts updated.")
        except Exception as exc:
            self.error(str(exc))

    def add_event(self,key):
        if not self.study or not self.study.active_run or self.process:
            return
        dialog = IntervalDialog(self.study,self.position,self.position+1,"Add spike interval" if key == "spikes" else "Mark possible seizure",True,self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        row = {"id":uuid.uuid4().hex,"start_sec":dialog.start.value(),"end_sec":dialog.end.value(),"channel":dialog.channel.currentText(),"status":"accepted","origin":"manual"}
        if key == "spikes":
            row.update(peak_sec=(row["start_sec"]+row["end_sec"])/2,amplitude_uv=np.nan,robust_z=np.nan)
        else:
            row["score"] = np.nan
        self.frames[key] = pd.concat([self.frames[key],pd.DataFrame([row])],ignore_index=True).sort_values("start_sec").reset_index(drop=True)
        self.populate_review()
        self.review_changed()

    def add_interval(self,kind):
        if not self.study or not self.study.active_run or self.process:
            return
        dialog = IntervalDialog(self.study,self.position,self.position+self.window_seconds.currentData(),"Exclude artifact" if kind == "artifact" else "Mark reviewed interval",parent=self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        span = {"start_sec":dialog.start.value(),"end_sec":dialog.end.value(),"reason":dialog.reason.text() or ("Artifact" if kind == "artifact" else "EEG interval reviewed"),"origin":"user"}
        (self.study.bad_spans if kind == "artifact" else self.study.reviewed_spans).append(span)
        self.refresh_intervals()
        self.review_changed()

    def remove_intervals(self):
        if not self.study or self.process:
            return
        selected = sorted({index.row() for index in self.interval_view.selectionModel().selectedRows()},reverse=True)
        for row in selected:
            item = self.interval_frame.iloc[row]
            if item.origin != "user":
                continue
            target = self.study.bad_spans if item.kind == "artifact" else self.study.reviewed_spans
            for index,span in enumerate(target):
                if span["start_sec"] == item.start_sec and span["end_sec"] == item.end_sec:
                    target.pop(index)
                    break
        self.refresh_intervals()
        self.review_changed()

    def save_notes(self):
        if self.study and not self.process:
            text = self.notes.toPlainText()
            if text != self.study.narrative:
                self.study.narrative = text
                self.store.save(self.study)

    def preview_report(self):
        if not self.study or not self.study.active_run:
            self.report_browser.setHtml("<h2>No analysis yet</h2><p>Check the electrodes, then run analysis.</p>")
            return
        self.save_notes()
        self.report_browser.setHtml(report_html(self.study,self.frames,include_charts=False))

    def import_eeg(self):
        path,_ = QtWidgets.QFileDialog.getOpenFileName(self,"Import EEG","","EEG signals (*.edf *.bdf *.fif *.vwr *.txt);;All files (*)")
        if path:
            self.start_job("import",path=path)

    def analyze(self):
        if not self.study:
            return
        self.save_notes()
        try:
            self.study.validate()
        except ValueError as exc:
            self.tabs.setCurrentIndex(2)
            self.error(str(exc))
            return
        self.start_job("analyze",study_id=self.study.id)

    def export(self):
        if not self.study or not self.study.active_run:
            return
        self.save_notes()
        destination = QtWidgets.QFileDialog.getExistingDirectory(self,"Choose report export folder",str(self.store.directory(self.study)))
        if destination:
            # Each export has its own directory, preserving previous reports.
            from datetime import datetime
            output = Path(destination) / ("EEG_report_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
            self.start_job("report",study_id=self.study.id,destination=str(output))

    def change_workspace(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self,"Choose local EEG workspace",str(self.store.workspace))
        if path:
            self.save_notes()
            if self.recording:
                self.recording.close()
            self.recording = self.study = self.frames = None
            self.store = Store(path)
            self.refresh_library()
            self.title.setText("Choose a recording")

    def start_job(self,action,**values):
        if self.process:
            return
        directory = self.store.root/"jobs"/uuid.uuid4().hex
        request = directory/"request.json"
        write_json(request,{"workspace":str(self.store.workspace),"action":action,**values})
        self.job_directory = directory
        self.job_action = action
        self.job_cancelled = False
        self.stdout_buffer = ""
        self.process = QtCore.QProcess(self)
        self.process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_job_output)
        self.process.finished.connect(self.job_finished)
        self.process.errorOccurred.connect(self.process_error)
        if getattr(sys,"frozen",False):
            arguments = ["--workspace",str(self.store.workspace),"--worker",str(request)]
        else:
            arguments = [str(Path(__file__).resolve().parents[1]/"launch_reviewer.py"),"--workspace",str(self.store.workspace),"--worker",str(request)]
        self.process.setProgram(sys.executable)
        self.process.setArguments(arguments)
        self.process.setWorkingDirectory(str(self.store.workspace))
        self.progress.show()
        self.progress.setValue(0)
        self.cancel_button.show()
        self.status.setText(f"Starting {action}…")
        self.update_buttons()
        self.process.start()
        self.job_poll_timer.start()

    def read_job_progress(self):
        if self.process:
            path = self.job_directory/"progress.json"
            if path.exists():
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                    self.progress.setValue(value["progress"])
                    self.status.setText(value["message"])
                except (OSError,ValueError,KeyError):
                    pass

    def read_job_output(self):
        if not self.process:
            return
        self.stdout_buffer += bytes(self.process.readAllStandardOutput()).decode("utf-8",errors="replace")
        while "\n" in self.stdout_buffer:
            line,self.stdout_buffer = self.stdout_buffer.split("\n",1)
            if line.startswith("EEG_REVIEWER "):
                try:
                    value = json.loads(line[len("EEG_REVIEWER "):])
                    if "progress" in value:
                        self.progress.setValue(value["progress"])
                        self.status.setText(value["message"])
                except ValueError:
                    pass
            else:
                with (self.job_directory/"worker.log").open("a",encoding="utf-8") as log:
                    log.write(line+"\n")

    def process_error(self,error):
        if error == QtCore.QProcess.ProcessError.FailedToStart:
            self.job_finished(-1)

    def job_finished(self,exit_code,*args):
        if not self.process:
            return
        self.read_job_output()
        self.process.deleteLater()
        self.process = None
        self.job_poll_timer.stop()
        self.progress.hide()
        self.cancel_button.hide()
        self.update_buttons()
        result_path = self.job_directory/"result.json"
        if self.job_cancelled:
            self.status.setText("Cancelled. Previous completed analysis and review edits are preserved.")
            self.refresh_library(self.study.id if self.study else None)
            return
        if not result_path.exists():
            self.error(f"The worker stopped (exit {exit_code}). Details: {self.job_directory / 'worker.log'}")
            return
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if not result.get("ok"):
            self.error(result.get("error","Analysis failed."))
            return
        self.refresh_library(result.get("study_id",self.study.id if self.study else None))
        if self.job_action == "report":
            try:
                from PySide6.QtPrintSupport import QPrinter
                html_path = Path(result["report"])
                document = QtGui.QTextDocument()
                document.setHtml(html_path.read_text(encoding="utf-8"))
                printer = QPrinter(QPrinter.PrinterMode.HighResolution)
                printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
                printer.setOutputFileName(str(html_path.with_suffix(".pdf")))
                printer.setPageSize(QtGui.QPageSize(QtGui.QPageSize.PageSizeId.A4))
                document.print_(printer)
                self.status.setText(f"PDF, HTML and CSV reports saved in {html_path.parent}")
                QtWidgets.QMessageBox.information(self,"Report exported",f"Saved PDF, HTML and analysis tables:\n{html_path.parent}")
            except Exception as exc:
                self.error(f"HTML and tables were exported, but PDF export failed: {exc}")
        elif self.job_action == "import":
            self.tabs.setCurrentIndex(2)
            self.status.setText("Recording imported. Verify the electrode mapping and save settings before analysis.")
        else:
            self.status.setText("Ready. Saved analyses loaded." if self.job_action == "discover" else "Analysis complete. Inspect stages and event candidates.")

    def cancel_job(self):
        if self.process:
            self.job_cancelled = True
            self.process.kill()

    def update_buttons(self):
        busy = self.process is not None
        self.library.setEnabled(not busy)
        self.import_button.setEnabled(not busy)
        self.workspace_button.setEnabled(not busy)
        self.tabs.setEnabled(not busy)
        self.analyze_button.setEnabled(bool(self.study) and not busy)
        self.export_button.setEnabled(bool(self.study and self.study.active_run and not self.study.settings_changed) and not busy)

    def error(self,message):
        self.status.setText(message)
        QtWidgets.QMessageBox.warning(self,"EEG Reviewer",message)

    def closeEvent(self,event):
        if self.process:
            choice = QtWidgets.QMessageBox.question(self,"Analysis running","Cancel the current job and close?",QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
            if choice != QtWidgets.QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.process.kill()
            self.process.waitForFinished(5000)
        self.save_notes()
        if self.recording:
            self.recording.close()
        event.accept()


STYLE = """
QWidget { font-family:'Segoe UI'; font-size:10pt; color:#243d52; }
QMainWindow, QWidget { background:#f5f7f9; }
QFrame#sidebar { background:#e8eff3; border-radius:12px; }
QLabel#brand { font-size:18pt; font-weight:700; color:#147d75; padding-top:12px; }
QLabel#title { font-size:22pt; font-weight:650; padding:12px 0; }
QLabel#privacy { color:#64778a; padding:12px 0; font-size:9pt; }
QLabel#notice { background:#fff0d7; border-radius:7px; padding:10px; color:#765624; }
QFrame#card { background:white; border:1px solid #dce5eb; border-radius:10px; padding:7px; }
QLabel#metric { font-size:20pt; font-weight:600; color:#147d75; }
QPushButton { background:white; border:1px solid #cbd7e0; padding:9px 13px; border-radius:6px; }
QPushButton:hover { background:#e6f2ef; border-color:#16877d; }
QPushButton[primary=true] { background:#16877d; color:white; border-color:#16877d; font-weight:600; }
QPushButton:disabled { background:#e6eaee; color:#9aa6b1; border-color:#e6eaee; }
QListWidget { border:0; background:transparent; }
QListWidget::item { padding:12px 7px; border-radius:5px; }
QListWidget::item:selected { background:#16877d; color:white; }
QTabWidget::pane { border:1px solid #dce5eb; background:white; border-radius:5px; }
QTabBar::tab { padding:10px 15px; background:#edf1f4; border:0; }
QTabBar::tab:selected { background:white; color:#147d75; font-weight:600; border-bottom:3px solid #16877d; }
QTableView, QTableWidget, QTextBrowser, QPlainTextEdit { background:white; border:1px solid #dce5eb; gridline-color:#e5ebf0; alternate-background-color:#f4f8fa; }
QHeaderView::section { background:#edf3f6; padding:7px; border:0; border-bottom:1px solid #d3dfe7; }
QLineEdit, QComboBox, QDoubleSpinBox { background:white; border:1px solid #cbd7e0; border-radius:4px; padding:5px; }
QProgressBar { border:1px solid #cbd7e0; border-radius:5px; text-align:center; }
QProgressBar::chunk { background:#16877d; }
"""


def configure_application(app):
    app.setApplicationName("EEG Reviewer")
    app.setOrganizationName("Local EEG Tools")
    app.setStyle("Fusion")
    style = STYLE
    if "Segoe UI" not in QtGui.QFontDatabase.families():
        import matplotlib
        font = Path(matplotlib.get_data_path())/"fonts/ttf/DejaVuSans.ttf"
        QtGui.QFontDatabase.addApplicationFont(str(font))
        app.setFont(QtGui.QFont("DejaVu Sans",10))
        style = style.replace("Segoe UI","DejaVu Sans")
    app.setStyleSheet(style)


def main(workspace):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    configure_application(app)
    window = MainWindow(workspace)
    window.show()
    return app.exec()


def smoke(workspace):
    """Check the packaged UI without requiring an interactive desktop."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    configure_application(app)
    output = Path(workspace)/"desktop_data"
    output.mkdir(parents=True,exist_ok=True)
    try:
        class SmokeWindow(MainWindow):
            errors = []
            def error(self,message):
                self.errors.append(message)
                self.status.setText(message)
        window = SmokeWindow(workspace)
        window.show()
        app.processEvents()
        checked = []
        for index in range(window.library.count()):
            window.library.setCurrentRow(index)
            app.processEvents()
            if window.errors:
                raise RuntimeError("; ".join(window.errors))
            if not window.study:
                raise RuntimeError("Could not load a prepared recording.")
            window.tabs.setCurrentIndex(1)
            window.show_eeg(60)
            app.processEvents()
            checked.append({"study":window.study.name,"spikes":len(window.frames["spikes"]),"swi":window.metrics["swi"]["percent"]})
        window.tabs.setCurrentIndex(0)
        app.processEvents()
        window.grab().save(str(output/"packaged_preview.png"))
        from .seizures import check_bundle
        model_directory = Store(workspace).models/"seizuretransformer"
        if not (model_directory/"manifest.json").exists() and getattr(sys,"frozen",False):
            model_directory = Path(sys._MEIPASS)/"models/seizuretransformer"
        manifest,model_path = check_bundle(model_directory)
        import torch
        torch.set_num_threads(2)
        with model_path.open("rb") as handle:
            model = torch.jit.load(handle,map_location="cpu").eval()
        with torch.inference_mode():
            predictions = model(torch.zeros(1,19,15360))
        if predictions.shape != (1,15360) or not torch.isfinite(predictions).all():
            raise RuntimeError("Packaged CPU model inference failed.")
        write_json(output/"smoke.json",{"ok":True,"studies":checked,"qt":QtCore.qVersion(),"cpu_model":"passed","progress_polling":hasattr(window,"job_poll_timer")})
        if window.process:
            window.process.kill()
            window.process.waitForFinished(5000)
        window.close()
        return 0
    except Exception as exc:
        import traceback
        write_json(output/"smoke.json",{"ok":False,"error":str(exc),"traceback":traceback.format_exc()})
        return 1
