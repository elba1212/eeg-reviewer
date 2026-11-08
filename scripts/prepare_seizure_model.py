"""Inspect/download the original competition artifact; never use third-party weights.

The --inspect command only reads registry metadata. --download extracts only
the original wu_2025 checkpoint/source/license files from the official image.
Export remains gated on a documented upstream parity check and distribution
terms; downloading an artifact alone does not enable the GUI detector.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tarfile
import urllib.request
import ssl
import sys


def open_url(request, timeout=60):
    import certifi
    # Use a standard CA bundle; the host Windows certificate store may contain
    # malformed entries. TLS verification remains enabled.
    return urllib.request.urlopen(request,timeout=timeout,context=ssl.create_default_context(cafile=certifi.where()))


REPOSITORY = "yujjio/seizure_transformer"


def registry_metadata():
    url = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{REPOSITORY}:pull"
    with open_url(url,timeout=60) as response:
        token = json.load(response)["token"]
    headers = {"Authorization":"Bearer " + token,
               "Accept":"application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json, application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json"}
    def manifest(reference):
        request = urllib.request.Request(f"https://registry-1.docker.io/v2/{REPOSITORY}/manifests/{reference}",headers=headers)
        with open_url(request,timeout=60) as response:
            return json.load(response), response.headers.get("Docker-Content-Digest",reference)
    document,digest = manifest("latest")
    if "manifests" in document:
        selected = next(item for item in document["manifests"] if item.get("platform",{}).get("architecture") == "amd64")
        document,digest = manifest(selected["digest"])
    return document,digest,headers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect",action="store_true")
    parser.add_argument("--download",action="store_true")
    parser.add_argument("--export",action="store_true",help="Verify official CPU inference and export a checked local TorchScript bundle")
    parser.add_argument("--destination",type=Path,default=Path(__file__).resolve().parents[1]/"desktop_data/models/upstream")
    parser.add_argument("--workspace",type=Path,default=None,help="Local application workspace for cache and model output")
    args = parser.parse_args()
    if args.export:
        export_bundle(args.destination,args.workspace)
        return
    document,digest,headers = registry_metadata()
    print(json.dumps({"official_image":REPOSITORY,"digest":digest,"layers":[{"digest":item["digest"],"compressed_mb":round(item["size"]/1e6,2)} for item in document["layers"]]},indent=2),flush=True)
    if not args.download:
        return
    args.destination.mkdir(parents=True,exist_ok=True)
    extracted = []
    # The final installation layer contains the competition Python package.
    for layer in reversed(document["layers"]):
        if layer["size"] < 1_000_000:
            continue
        print(f"Inspecting package layer ({layer['size']/1e6:.1f} MB compressed)",flush=True)
        request = urllib.request.Request(f"https://registry-1.docker.io/v2/{REPOSITORY}/blobs/{layer['digest']}",headers=headers)
        with open_url(request,timeout=180) as response, tarfile.open(fileobj=response,mode="r|gz") as archive:
            for member in archive:
                if not member.isfile() or "wu_2025" not in member.name:
                    continue
                relative = Path(member.name)
                # Flat extraction prevents archive paths from escaping destination.
                if relative.suffix in {".pth",".pt",".py"} or relative.name.lower().startswith(("license","copying","metadata")):
                    target = args.destination/relative.name
                    stream = archive.extractfile(member)
                    if stream:
                        with target.open("wb") as handle:
                            while data := stream.read(1024*1024):
                                handle.write(data)
                        print(f"Extracted {relative.name} ({member.size/1e6:.1f} MB)",flush=True)
                        extracted.append(relative.name)
            if any(name.endswith((".pth",".pt")) for name in extracted):
                break
    provenance = {"official_image":REPOSITORY,"digest":digest,"extracted":extracted,
                  "source_url":"https://github.com/keruiwu/SeizureTransformer",
                  "status":"Downloaded only; distribution terms and upstream inference parity require verification."}
    (args.destination/"provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    if not any(name.endswith((".pth",".pt")) for name in extracted):
        raise RuntimeError("No official checkpoint found. Screening remains unavailable.")


def export_bundle(source, workspace=None):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root))
    if workspace is not None:
        sys.argv.extend(["--workspace",str(workspace)])
    import launch_reviewer
    launch_reviewer.bootstrap()
    import hashlib
    import importlib
    import types
    import numpy as np
    from scipy.signal import resample
    import torch
    from eeg_reviewer.seizures import MODEL_CHANNELS, preprocess_window
    from eeg_reviewer.storage import write_json
    torch.set_num_threads(2)
    source = source.resolve()
    provenance = json.loads((source/"provenance.json").read_text(encoding="utf-8"))
    if provenance["official_image"] != REPOSITORY:
        raise ValueError("Unexpected checkpoint provenance")
    license_url = "https://raw.githubusercontent.com/keruiwu/SeizureTransformer/main/LICENSE"
    license_path = source/"LICENSE"
    if not license_path.exists():
        with open_url(license_url) as response:
            license_path.write_bytes(response.read())
    if "MIT License" not in license_path.read_text(encoding="utf-8"):
        raise ValueError("Upstream license has changed; review distribution terms before export.")
    package = types.ModuleType("wu_2025")
    package.__path__ = [str(source)]
    sys.modules["wu_2025"] = package
    architecture = importlib.import_module("wu_2025.architecture")
    utils = importlib.import_module("wu_2025.utils")
    print("Loading original CPU checkpoint",flush=True)
    reference = architecture.SeizureTransformer().eval()
    checkpoint = torch.load(source/"model.pth",map_location="cpu",weights_only=True)
    reference.load_state_dict(checkpoint,strict=True)
    del checkpoint
    rng = np.random.default_rng(20261108)
    tests = []
    preprocess_error = 0.0
    for fs in [200,256,500]:
        raw = rng.standard_normal((19,int(fs*67.25)))
        loader = utils.get_dataloader(raw,fs,batch_size=1)
        normalized = (raw-raw.mean(axis=1,keepdims=True))/raw.std(axis=1,keepdims=True)
        prepared = resample(normalized,int(raw.shape[1]*256/fs),axis=1) if fs != 256 else normalized
        for index,expected in enumerate(loader):
            actual = preprocess_window(prepared[:,index*15360:(index+1)*15360])
            error = float(np.max(np.abs(actual-expected.numpy()[0])))
            preprocess_error = max(preprocess_error,error)
            if not np.allclose(actual,expected.numpy()[0],atol=1e-6,rtol=1e-6):
                raise ValueError(f"Preprocessing parity failed for {fs} Hz: {error}")
            if index == 0:
                tests.append(torch.from_numpy(actual)[None])
    print(f"Upstream preprocessing parity passed; max error={preprocess_error:g}",flush=True)
    with torch.inference_mode():
        traced = torch.jit.trace(reference,tests[0],check_trace=False)
        traced = torch.jit.freeze(traced.eval())
        maximum = 0.0
        for test in tests:
            expected = reference(test)
            actual = traced(test)
            error = float((actual-expected).abs().max())
            maximum = max(maximum,error)
            if not torch.allclose(actual,expected,atol=1e-5,rtol=1e-5):
                raise ValueError(f"CPU inference parity failed: {error}")
    destination = source.parent/"seizuretransformer"
    destination.mkdir(parents=True,exist_ok=True)
    output = destination/"model.pt"
    with output.open("wb") as handle:
        torch.jit.save(traced,handle)
    (destination/"LICENSE").write_bytes(license_path.read_bytes())
    parity = {"preprocessing_max_abs_error":preprocess_error,"inference_max_abs_error":maximum,
              "sample_rates_tested":[200,256,500],"windows":"60 s and padded final window",
              "scope":"Numerical preprocessing and CPU model parity, not clinical validation.","torch_version":torch.__version__}
    write_json(destination/"parity.json",parity)
    manifest = {"model":"SeizureTransformer","format":"torchscript","source_url":"https://github.com/keruiwu/SeizureTransformer",
                "official_image":REPOSITORY,"official_image_digest":provenance["digest"],"distribution_terms":"Upstream MIT License; original competition artifact; see bundled LICENSE and source provenance.",
                "parity_passed":True,"sha256":hashlib.sha256(output.read_bytes()).hexdigest(),"fs":256,"window_samples":15360,
                "montage":"referential","channels":MODEL_CHANNELS,"threshold":0.8,"clinical_validation":"Not established for pediatric epilepsy recordings.",
                "timestamp_correction":"The original wrapper truncates/rescores non-256-Hz output using original sample count. This app uses 256-Hz output times and clips to actual duration."}
    write_json(destination/"manifest.json",manifest)
    print(json.dumps({"bundle":str(destination),**parity},indent=2),flush=True)


if __name__ == "__main__":
    main()
