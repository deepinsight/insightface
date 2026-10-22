"""Shared button tooltip helpers."""

from __future__ import annotations

import re

from PySide6.QtWidgets import QAbstractButton, QWidget


BUTTON_TOOLTIPS = {
    "Add Folder": "Add an album folder to the import list.",
    "Add Person": "Create a new person entry in the local People Library.",
    "Add as New Person": "Register the current face as a new person.",
    "Apply": "Apply changes without closing this dialog.",
    "Assign to Existing Person": "Assign the current face to an existing person.",
    "Browse": "Choose a local file or folder.",
    "Cancel": "Close without applying changes.",
    "Check Cameras": "Check available local camera devices.",
    "Clear": "Clear the current selection or form.",
    "Compare": "Run 1:1 face comparison for the selected images.",
    "Contact Enterprise Support": "Open the InsightFace enterprise support contact page.",
    "Delete": "Delete the selected item.",
    "Delete Selected": "Delete the selected record.",
    "Download Selected": "Download the selected model package to the local model cache.",
    "Exit": "Close the application.",
    "Export": "Export the current results.",
    "Export All Summary": "Export a summary of all available records.",
    "Export Annotated Image": "Export the image with face labels drawn on it.",
    "Export CSV": "Export the current table as CSV.",
    "Export CSV / JSON": "Export current results as CSV and JSON.",
    "Export License Summary": "Save a local license summary report.",
    "Export Report": "Export the current evaluation report.",
    "Export Result": "Export the current result.",
    "Export Results": "Export the current results.",
    "Export Settings": "Export application settings to a JSON file.",
    "Import / Refresh": "Scan selected folders for new images and rebuild album clusters from indexed faces.",
    "Import Image or Folder": "Import one image or a folder of images by clicking or dragging files here.",
    "Import Settings": "Import application settings from a JSON file.",
    "Open": "Open the selected item.",
    "Open GitHub Releases": "Open the InsightFace GitHub Releases page.",
    "Open License Center": "Open license and commercial usage information.",
    "Open Model Downloads": "Open the model downloads tab.",
    "Open Model Folder": "Open the local InsightFace model folder.",
    "Open Report Folder": "Open the local folder where reports are saved.",
    "Open Result Directory": "Open the local folder where face swap results are saved.",
    "Open Selected Report": "Open the selected report file.",
    "Open Workspace Folder": "Open the local workspace folder.",
    "Re-export Selected": "Regenerate and export the selected report.",
    "Rebuild All": "Reprocess selected album folders from scratch after confirmation.",
    "Recognize Faces": "Detect and recognize all faces in the selected image.",
    "Refresh": "Reload the current list from local storage.",
    "Refresh Download URLs": "Refresh available model download URLs from GitHub Releases.",
    "Refresh People": "Reload people from the local People Library.",
    "Register Current Face": "Register the current camera face as a person.",
    "Remove": "Remove the selected item.",
    "Remove Selected": "Remove selected items from this list.",
    "Request Commercial Model License": "Show contact options for commercial model licensing.",
    "Request Face Swap Commercial License": "Show contact options for face swap commercial licensing.",
    "Request Private Model Evaluation": "Show contact options for private model evaluation.",
    "Request SDK / API": "Show contact options for SDK or API access.",
    "Reset": "Reset this page to its initial state.",
    "Reset Settings": "Reset application settings to their defaults.",
    "Run Evaluation": "Run the selected local enterprise evaluation.",
    "Run Face Swap": "Run face swap with the configured local swap model.",
    "Run Recognition": "Run 1:1 comparison or 1:N gallery recognition.",
    "Save": "Save changes and close this dialog.",
    "Save Details": "Save the current person details.",
    "Save Settings": "Save the current settings.",
    "Scan": "Scan selected local data.",
    "Scan and Cluster": "Scan images and cluster detected faces.",
    "Search": "Search the local People Library for matching faces.",
    "Select": "Select a local file or folder.",
    "Start": "Start the current processing task.",
    "Stop": "Stop the current processing task.",
    "Test Model Load": "Test loading the selected model and runtime provider.",
    "Use Selected Model": "Use the selected downloaded model package.",
    "Visit Homepage": "Open the InsightFace homepage.",
    "Warmup": "Run a short model warmup to measure startup latency.",
    "×": "Remove the current file.",
}


def normalize_button_text(text: str) -> str:
    value = str(text or "").replace("&", "").strip()
    value = re.sub(r"\s+", " ", value)
    return value


def tooltip_for_button(text: str) -> str:
    label = normalize_button_text(text)
    if not label:
        return "Click this button to run the related action."
    if label in BUTTON_TOOLTIPS:
        return BUTTON_TOOLTIPS[label]
    if label.lower() == "ok":
        return "Confirm and continue."
    return f"Click to {label[0].lower() + label[1:]}."


def set_button_tooltip(button: QAbstractButton, tooltip: str | None = None) -> None:
    if button.toolTip():
        return
    button.setToolTip(tooltip or tooltip_for_button(button.text()))


def apply_button_tooltips(widget: QWidget) -> None:
    for button in widget.findChildren(QAbstractButton):
        set_button_tooltip(button)
