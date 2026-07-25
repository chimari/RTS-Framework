"""Integration test for Step 05 directory batch mode v5.11.0."""
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
        rows.append({"dataset":"bias","directory":str(root),"environment":"step05-v5.11-test",
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


def write_target(path: Path):
    fits.PrimaryHDU(data=np.zeros((2,3),dtype=np.uint16)).writeto(path)


def call_cli(args):
    out,err=io.StringIO(),io.StringIO()
    with redirect_stdout(out),redirect_stderr(err):
        code=step05.run_rts_correction_batch_cli(args)
    return code,out.getvalue(),err.getvalue()


def main():
    print("="*72); print("RTS Framework Step 05 directory batch test"); print("="*72)
    print(f"step05 version : {step05.__version__}\n")
    with tempfile.TemporaryDirectory(prefix="rts_step05_directory_") as temp:
        root=Path(temp); metadata=prepare(root)
        first=root/"science_a"; second=root/"science_b"; first.mkdir(); second.mkdir()
        write_target(first/"zeta.fit"); write_target(first/"alpha.fit")
        write_target(first/"ignored.fits")
        write_target(second/"beta.fit")
        (first/"notes.txt").write_text("ignore",encoding="utf-8")

        print("[1/4] Directory discovery is deterministic and filtered")
        found=step05.discover_rts_batch_inputs([first,second],pattern="*.fit")
        require([p.name for p in found]==["alpha.fit","zeta.fit","beta.fit"],"discovery order changed")
        require(all(p.is_absolute() for p in found),"paths are not normalized")
        print("   Pattern       : *.fit\n   Ordering      : directory order + filename sort\n   Non-matches   : ignored\n   Result        : PASS\n")

        print("[2/4] Invalid discovery inputs fail safely")
        try: step05.discover_rts_batch_inputs([root/"missing"],pattern="*.fit"); require(False,"missing directory accepted")
        except step05.Step05Error: pass
        empty=root/"empty"; empty.mkdir()
        try: step05.discover_rts_batch_inputs([empty],pattern="*.fit"); require(False,"empty match accepted")
        except step05.Step05Error: pass
        try: step05.discover_rts_batch_inputs([first,first],pattern="*.fit"); require(False,"duplicates accepted")
        except step05.Step05Error: pass
        print("   Missing dir   : rejected\n   Zero matches  : rejected\n   Duplicates    : rejected\n   Result        : PASS\n")

        print("[3/4] CLI preflight accepts directory inputs")
        outdir=root/"preflight_outputs"; outdir.mkdir()
        args=["--metadata",str(metadata),"--input-dir",str(first),"--pattern","*.fit",
              "--output-directory",str(outdir),"--preflight","--json"]
        code,a,err=call_cli(args); code2,b,err2=call_cli(args)
        require(code==code2==0,"directory preflight failed")
        require(err==err2=="","unexpected stderr")
        require(a==b,"directory JSON is not deterministic")
        payload=json.loads(a); require(payload["total_count"]==2 and payload["is_valid"] is True,"preflight count changed")
        require(not any(outdir.iterdir()),"preflight wrote files")
        print("   Inputs found  : 2\n   JSON output   : deterministic\n   Output writes : none\n   Result        : PASS\n")

        print("[4/4] CLI corrects directory inputs and composes sources")
        corr=root/"corrected"; corr.mkdir()
        explicit=second/"beta.fit"
        args=["--metadata",str(metadata),"--input",str(explicit),"--input-dir",str(first),
              "--pattern","*.fit","--output-directory",str(corr),"--json"]
        code,text,err=call_cli(args)
        require(code==0 and err=="","directory correction failed")
        payload=json.loads(text); require(payload["total_count"]==3 and payload["succeeded_count"]==3,"combined count changed")
        expected={"beta_rts_corrected.fit","alpha_rts_corrected.fit","zeta_rts_corrected.fit"}
        require({p.name for p in corr.iterdir()}==expected,"corrected outputs changed")
        bad_code,bad_out,bad_err=call_cli(["--metadata",str(metadata),"--pattern","*.fit",
             "--output-directory",str(corr),"--quiet"])
        require(bad_code==1 and bad_out==bad_err=="","standalone pattern handling changed")
        print("   Explicit + dir: supported\n   Corrected      : 3\n   Pattern alone  : rejected\n   Result         : PASS\n")
    print("="*72); print("FINISHED: Step 05 directory batch test passed"); print("="*72); return 0

if __name__=="__main__": raise SystemExit(main())
