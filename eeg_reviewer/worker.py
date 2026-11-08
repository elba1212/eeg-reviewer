"""JSON-line process protocol used by the desktop app and CLI."""
import json
from pathlib import Path
import traceback

from .storage import Store, clean_json, write_json


def main(request_path):
    request_path = Path(request_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    store = Store(request["workspace"])
    def progress(percent, message):
        value = {"progress":percent,"message":message}
        write_json(request_path.parent/"progress.json",value)
        print("EEG_REVIEWER " + json.dumps(value), flush=True)
    try:
        action = request["action"]
        if action == "import":
            from .recordings import import_recording
            study = import_recording(request["path"], store, progress)
            result = {"ok": True, "study_id": study.id}
        elif action == "analyze":
            from .pipeline import run_analysis
            study = run_analysis(store, store.load(request["study_id"]), progress, request.get("run_seizures", True))
            result = {"ok": True, "study_id": study.id}
        elif action == "report":
            from .reporting import export_report
            study = store.load(request["study_id"])
            output = export_report(store, study, request["destination"])
            result = {"ok": True, "study_id": study.id, "report": str(output)}
        else:
            raise ValueError(f"Unknown job: {action}")
    except Exception as exc:
        (request_path.parent / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        result = {"ok": False, "error": str(exc)}
    write_json(request_path.parent / "result.json", result)
    print("EEG_REVIEWER " + json.dumps(clean_json(result)), flush=True)
    return 0 if result["ok"] else 1
