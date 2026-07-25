"""Integration test for Step 05 batch preflight validation v5.10.0."""
from __future__ import annotations
import csv, io, json, sys, tempfile
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import FrozenInstanceError
from pathlib import Path
import numpy as np
from astropy.io import fits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from steps import step03_prepare_bias_analysis as step03
from steps import step04_prepare_rts_dictionary_analysis as step04
from steps import step05_apply_rts_correction as step05

def require(condition, message):
    if not condition:
        print(f"FAIL: {message}")
        raise SystemExit(1)

def prepare(root: Path):
    files=[]
    for i in range(8):
        data=np.array([[0 if i%4<2 else 10,20,30],[10 if i%4<2 else 0,5,50]],dtype=np.uint16)
        p=root/f"bias_{i:04d}.fit"; fits.PrimaryHDU(data=data).writeto(p); files.append(p)
    rows=[]
    for i,p in enumerate(files):
        rows.append({"dataset":"bias","directory":str(root),"environment":"step05-v5.10-test",
        "frame_index":i,"n_frames":8,"temperature_C":-10.0,"temperature_start_C":-10.0,
        "temperature_end_C":-10.0,"temperature_fraction":i/7,"exposure_s":0.0,
        "filename":p.name,"filepath":str(p),"image_width":3,"image_height":2,
        "pixel_dtype":"uint16","byte_order":"not-applicable"})
    m=root/"manifest.normalized.csv"
    with m.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n"); w.writeheader(); w.writerows(rows)
    analysis=step03.prepare_bias_analysis(m,"bias")
    built=step04.build_rts_dictionary_artifacts(step04.prepare_rts_dictionary_analysis(analysis),root/"dictionary.csv",
        minimum_score=0.9,minimum_state_count=2,minimum_separation=5.0,
        minimum_transition_count=3,minimum_lower_run=2,minimum_upper_run=2)
    step04.audit_rts_dictionary_input_files(built.metadata_path)
    valid=[]
    for i in range(2):
        p=root/f"target_{i}.fit"; fits.PrimaryHDU(data=np.zeros((2,3),dtype=np.uint16)).writeto(p); valid.append(p)
    bad=root/"bad_shape.fit"; fits.PrimaryHDU(data=np.zeros((3,3),dtype=np.uint16)).writeto(bad)
    return built.metadata_path, tuple(valid), bad

def call_cli(args):
    out,err=io.StringIO(),io.StringIO()
    with redirect_stdout(out),redirect_stderr(err):
        code=step05.run_rts_correction_batch_cli(args)
    return code,out.getvalue(),err.getvalue()

def main():
    print("="*72); print("RTS Framework Step 05 batch preflight test"); print("="*72)
    print(f"step05 version : {step05.__version__}\n")
    with tempfile.TemporaryDirectory(prefix="rts_step05_preflight_") as temp:
        root=Path(temp); metadata,valid,bad=prepare(root); outdir=root/"outputs"; outdir.mkdir()
        print("[1/4] Valid batch passes preflight")
        result=step05.validate_rts_correction_batch(metadata,valid,outdir)
        require(result.is_valid,"valid batch rejected"); require(result.error_count==0,"unexpected errors")
        require(result.valid_item_count==2,"valid count changed")
        require(not any(outdir.iterdir()),"preflight wrote output artifacts")
        try: result.overwrite=True; require(False,"result is mutable")
        except FrozenInstanceError: pass
        print("   Inputs        : 2\n   Artifacts     : valid\n   Output writes : none\n   Result        : PASS\n")

        print("[2/4] Missing, duplicate, and mismatched inputs are reported")
        missing=root/"missing.fit"
        invalid=step05.validate_rts_correction_batch(metadata,[valid[0],valid[0],missing,bad],outdir)
        codes=[issue.code for item in invalid.items for issue in item.issues]
        require("DUPLICATE_INPUT" in codes,"duplicate not reported")
        require("INVALID_INPUT" in codes,"missing input not reported")
        require("SHAPE_MISMATCH" in codes,"shape mismatch not reported")
        require(not invalid.is_valid,"invalid batch accepted")
        print("   Duplicate     : reported\n   Missing       : reported\n   Shape mismatch: reported\n   Result        : PASS\n")

        print("[3/4] Output collisions and overwrite rules are enforced")
        existing=outdir/f"{valid[0].stem}_rts_corrected{valid[0].suffix}"; existing.write_bytes(b"protected")
        collision=step05.validate_rts_correction_batch(metadata,[valid[0]],outdir)
        require(collision.items[0].issues[0].code=="OUTPUT_EXISTS","existing output not protected")
        allowed=step05.validate_rts_correction_batch(metadata,[valid[0]],outdir,overwrite=True)
        require(allowed.is_valid,"overwrite=True should permit existing output")
        require(existing.read_bytes()==b"protected","preflight modified existing output")
        print("   Existing file : protected\n   Overwrite flag : respected\n   File contents  : unchanged\n   Result         : PASS\n")

        print("[4/4] CLI text and JSON outputs are deterministic")
        args=["--metadata",str(metadata),"--input",str(valid[1]),"--output-directory",str(outdir),"--preflight","--json"]
        code,a,err=call_cli(args); code2,b,err2=call_cli(args)
        require(code==0 and code2==0,"valid CLI preflight failed")
        require(err==err2=="","unexpected stderr")
        require(a==b,"JSON output is not deterministic")
        payload=json.loads(a); require(payload["status"]=="OK" and payload["is_valid"] is True,"JSON payload changed")
        require(not (outdir/f"{valid[1].stem}_rts_corrected{valid[1].suffix}").exists(),"CLI wrote corrected FITS")
        bad_args=["--metadata",str(metadata),"--input",str(bad),"--output-directory",str(outdir),"--preflight","--quiet"]
        bad_code,bad_out,bad_err=call_cli(bad_args)
        require(bad_code==1 and bad_out==bad_err=="","quiet invalid CLI changed")
        print("   JSON output   : deterministic\n   Invalid exit  : 1\n   Quiet mode    : silent\n   Result        : PASS\n")
    print("="*72); print("FINISHED: Step 05 batch preflight test passed"); print("="*72); return 0
if __name__=="__main__": raise SystemExit(main())
