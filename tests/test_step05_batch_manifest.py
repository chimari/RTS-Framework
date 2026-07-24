"""Integration test for Step 05 batch manifests v5.8.0."""
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

def source_manifest(root):
    files=[]
    for i in range(8):
        data=np.array([[0 if i%4<2 else 10,20,30],
                       [10 if i%4<2 else 0,5,50]],dtype=np.uint16)
        p=root/f"bias_{i:04d}.fit"; fits.PrimaryHDU(data=data).writeto(p); files.append(p)
    rows=[]
    for i,p in enumerate(files):
        rows.append({"dataset":"bias","directory":str(root),
        "environment":"step05-v5.8-test","frame_index":i,"n_frames":8,
        "temperature_C":-10.0,"temperature_start_C":-10.0,
        "temperature_end_C":-10.0,"temperature_fraction":i/7,
        "exposure_s":0.0,"filename":p.name,"filepath":str(p),
        "image_width":3,"image_height":2,"pixel_dtype":"uint16",
        "byte_order":"not-applicable"})
    m=root/"manifest.normalized.csv"
    with m.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    return m

def prepare(root):
    a=step03.prepare_bias_analysis(source_manifest(root),"bias")
    b=step04.build_rts_dictionary_artifacts(
        step04.prepare_rts_dictionary_analysis(a),root/"dictionary.csv",
        minimum_score=0.9,minimum_state_count=2,minimum_separation=5.0,
        minimum_transition_count=3,minimum_lower_run=2,minimum_upper_run=2)
    step04.audit_rts_dictionary_input_files(b.metadata_path)
    inputs=[]
    for j in range(2):
        p=root/f"target_{j}.fit"; d=np.full((2,3),20+j,dtype=np.float32)
        fits.PrimaryHDU(data=d).writeto(p)
        plan=step05.prepare_rts_correction(b.metadata_path,p)
        for i,c in enumerate(plan.candidates):
            d[c.row,c.column]=(c.lower_state_center+0.1*c.state_separation
                if (i+j)%2==0 else c.upper_state_center-0.1*c.state_separation)
        fits.PrimaryHDU(data=d).writeto(p,overwrite=True); inputs.append(p)
    bad=root/"invalid.fit"; fits.PrimaryHDU(data=np.zeros((3,3))).writeto(bad)
    return b.metadata_path,tuple(inputs),bad

def cli(args):
    out,err=io.StringIO(),io.StringIO()
    with redirect_stdout(out),redirect_stderr(err):
        code=step05.run_rts_correction_batch_cli(args)
    return code,out.getvalue(),err.getvalue()

def main():
    print("="*72); print("RTS Framework Step 05 batch manifest test"); print("="*72)
    print(f"step05 version : {step05.__version__}\n")
    with tempfile.TemporaryDirectory(prefix="rts_step05_manifest_") as t:
        root=Path(t); metadata,inputs,bad=prepare(root)
        outdir=root/"outputs"; outdir.mkdir()
        result=step05.run_rts_correction_batch(metadata,inputs,outdir)

        print("[1/4] JSON manifest is deterministic and complete")
        jp=root/"batch.json"; step05.write_rts_batch_manifest_json(result,jp)
        p=json.loads(jp.read_text())
        require(p["manifest_version"]==1,"version missing")
        require(p["total_count"]==2,"total changed")
        require(all(len(x["output"]["sha256"])==64 for x in p["items"]),"SHA missing")
        first=jp.read_bytes(); step05.write_rts_batch_manifest_json(result,jp,overwrite=True)
        require(first==jp.read_bytes(),"JSON not byte-stable")
        print("   Records       : 2\n   SHA256        : included\n   Rewrite       : byte-identical\n   Result        : PASS\n")

        print("[2/4] CSV manifest records correction metrics")
        cp=root/"batch.csv"; step05.write_rts_batch_manifest_csv(result,cp)
        with cp.open(newline="") as f: rows=list(csv.DictReader(f))
        require(len(rows)==2,"row count")
        require(rows[0]["applied_count"]!="","applied missing")
        require(rows[0]["preserved_count"]!="","preserved missing")
        require(rows[0]["verified"]=="True","verified missing")
        require(len(rows[0]["sha256"])==64,"SHA missing")
        print("   Rows          : 2\n   Metrics       : included\n   Verified      : True\n   Result        : PASS\n")

        print("[3/4] Partial manifests preserve failure details")
        pd=root/"partial_outputs"; pd.mkdir()
        partial=step05.run_rts_correction_batch(metadata,[inputs[0],bad,inputs[1]],pd,continue_on_error=True)
        pjp,pcp=root/"partial.json",root/"partial.csv"
        step05.write_rts_batch_manifest_json(partial,pjp)
        step05.write_rts_batch_manifest_csv(partial,pcp)
        pj=json.loads(pjp.read_text()); require(pj["failed_count"]==1,"failure count")
        require(pj["items"][1]["error"],"JSON error")
        with pcp.open(newline="") as f: pr=list(csv.DictReader(f))
        require(pr[1]["succeeded"]=="False","CSV failure")
        require(pr[1]["error"],"CSV error"); require(pr[1]["sha256"]=="","failed SHA")
        print("   Successes     : 2\n   Failures      : 1\n   Error details : preserved\n   Result        : PASS\n")

        print("[4/4] CLI writes manifests and protects existing files")
        cd=root/"cli_outputs"; cd.mkdir(); cj,cc=root/"cli.json",root/"cli.csv"
        code,stdout,stderr=cli(["--metadata",str(metadata),"--input",str(inputs[0]),
        "--input",str(inputs[1]),"--output-directory",str(cd),
        "--manifest-json",str(cj),"--manifest-csv",str(cc),"--json"])
        require(code==0,f"CLI failed {code}"); require(stderr=="","stderr")
        require(cj.exists() and cc.exists(),"manifest missing")
        require(json.loads(stdout)["status"]=="OK","status")
        protected=root/"protected"; protected.mkdir()
        code,stdout,stderr=cli(["--metadata",str(metadata),"--input",str(inputs[0]),
        "--output-directory",str(protected),"--manifest-json",str(cj)])
        require(code==1,"existing overwritten"); require("ERROR:" in stderr,"error missing")
        print("   JSON manifest : written\n   CSV manifest  : written\n   Existing file : protected\n   Result        : PASS\n")
    print("="*72); print("FINISHED: Step 05 batch manifest test passed"); print("="*72)
    return 0
if __name__=="__main__": raise SystemExit(main())
