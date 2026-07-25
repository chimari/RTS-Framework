"""Integration test for Step 05 multiprocessing batch mode v5.12.0."""
from __future__ import annotations
import csv, io, json, sys, tempfile
from contextlib import redirect_stderr, redirect_stdout
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
        rows.append({"dataset":"bias","directory":str(root),"environment":"step05-v5.12-test",
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
    return built.metadata_path


def write_target(path: Path, shape=(2,3)):
    fits.PrimaryHDU(data=np.zeros(shape,dtype=np.uint16)).writeto(path)


def call_cli(args):
    out,err=io.StringIO(),io.StringIO()
    with redirect_stdout(out),redirect_stderr(err):
        code=step05.run_rts_correction_batch_cli(args)
    return code,out.getvalue(),err.getvalue()


def main():
    print("="*72); print("RTS Framework Step 05 multiprocessing batch test"); print("="*72)
    print(f"step05 version : {step05.__version__}\n")
    with tempfile.TemporaryDirectory(prefix="rts_step05_parallel_") as temp:
        root=Path(temp); metadata=prepare(root)
        inputs=[]
        for name in ("gamma.fit","alpha.fit","beta.fit"):
            p=root/name; write_target(p); inputs.append(p)

        print("[1/4] Parallel batch preserves input order")
        outdir=root/"parallel"; outdir.mkdir()
        result=step05.run_rts_correction_batch(metadata,inputs,outdir,
            continue_on_error=True,workers=2)
        require(result.all_succeeded,"parallel batch failed")
        require([item.input_path.name for item in result.items]==[p.name for p in inputs],"input order changed")
        require([item.output_path.name for item in result.items]==["gamma_rts_corrected.fit","alpha_rts_corrected.fit","beta_rts_corrected.fit"],"output order changed")
        print("   Workers       : 2\n   Result order  : input order\n   Outputs       : 3\n   Result        : PASS\n")

        print("[2/4] Parallel failure is recorded deterministically")
        bad=root/"bad.fit"; write_target(bad,shape=(1,1))
        outdir2=root/"partial"; outdir2.mkdir()
        partial=step05.run_rts_correction_batch(metadata,[inputs[0],bad,inputs[1]],outdir2,
            continue_on_error=True,workers=2)
        require([item.succeeded for item in partial.items]==[True,False,True],"failure ordering changed")
        require(partial.failed_count==1 and partial.succeeded_count==2,"failure counts changed")
        require(partial.items[1].error,"failure detail missing")
        print("   Successes     : 2\n   Failures      : 1\n   Item order    : preserved\n   Result        : PASS\n")

        print("[3/4] Unsafe and invalid worker settings are rejected")
        try:
            step05.run_rts_correction_batch(metadata,inputs,root/"serial_guard",workers=0)
            require(False,"workers=0 accepted")
        except step05.Step05Error: pass
        guard=root/"guard"; guard.mkdir()
        try:
            step05.run_rts_correction_batch(metadata,inputs,guard,workers=2,continue_on_error=False)
            require(False,"parallel fail-fast accepted")
        except step05.Step05Error: pass
        print("   workers=0     : rejected\n   Parallel fail-fast: rejected\n   Safety rule   : explicit\n   Result        : PASS\n")

        print("[4/4] CLI parallel mode and parent aggregation are deterministic")
        cliout=root/"cli"; cliout.mkdir()
        manifest=root/"batch.json"; provenance=root/"provenance.json"
        args=["--metadata",str(metadata),"--input",str(inputs[0]),"--input",str(inputs[1]),
              "--output-directory",str(cliout),"--workers","2","--continue-on-error",
              "--manifest-json",str(manifest),"--provenance-json",str(provenance),"--json"]
        code,text,err=call_cli(args)
        require(code==0 and err=="","parallel CLI failed")
        payload=json.loads(text)
        require(payload["workers"]==2 and payload["succeeded_count"]==2,"CLI summary changed")
        require(manifest.exists() and provenance.exists(),"parent artifacts missing")
        manifest_payload=json.loads(manifest.read_text(encoding="utf-8"))
        require([Path(item["input_path"]).name for item in manifest_payload["items"]]==[inputs[0].name,inputs[1].name],"manifest order changed")
        bad_code,bad_out,bad_err=call_cli(["--metadata",str(metadata),"--input",str(inputs[0]),
            "--output-directory",str(root/"bad_cli"),"--workers","2","--quiet"])
        require(bad_code==1 and bad_out==bad_err=="","CLI safety rejection changed")
        print("   CLI workers   : 2\n   Manifest      : parent aggregated\n   Provenance    : parent aggregated\n   Result        : PASS\n")
    print("="*72); print("FINISHED: Step 05 multiprocessing batch test passed"); print("="*72); return 0

if __name__=="__main__": raise SystemExit(main())
