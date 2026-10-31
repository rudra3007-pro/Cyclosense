#!/usr/bin/env python3
"""
CycloSense - Live Model Evaluation CLI
=======================================
Loads actual model weights and runs REAL inference every time.
Optionally accepts test data paths for full metric computation.

Usage:
  python evaluate.py                                       # All models, live inference
  python evaluate.py --model 2                             # Model 2 only
  python evaluate.py --model 2 --m2-csv labels.csv --m2-img-dir imgs/
  python evaluate.py --m1-data path/to/data.yaml
  python evaluate.py --m3-data ibtracs.csv
  python evaluate.py --json --export results.json
"""

import sys, os, time, json, math, argparse, pickle
from datetime import datetime

import torch
import torchvision.models as tv_models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

# Path setup
CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
BACKEND_DIR  = os.path.join(PROJECT_ROOT, "backend")
WEIGHTS_DIR  = os.path.join(BACKEND_DIR, "weights")
for p in [PROJECT_ROOT, BACKEND_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

CLASSES_M2  = ["CS", "D_DD", "VSCS"]   # sorted alphabetically — must match training
NUM_CLASSES = len(CLASSES_M2)
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CycloneLSTM(torch.nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, output_size=3):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc   = torch.nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

W = 80
def _bar(c="="): return c * W
def _sub():      return "-" * W
def _pct(v):     return f"{v*100:.1f}%"
def _ok(v):      return "\033[92mPASS\033[0m" if v else "\033[91mFAIL\033[0m"
def _header(t):  print(); print(_bar()); print(f"  {t}"); print(_bar())
def _section(t): print(); print(_sub()); print(f"  {t}"); print(_sub())

def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

# Loaders
def _load_yolo():
    from ultralytics import YOLO
    path = os.path.join(WEIGHTS_DIR, "model1_detection.pt")
    if not os.path.exists(path): raise FileNotFoundError(path)
    return YOLO(path), path

def _load_effnet():
    path = os.path.join(WEIGHTS_DIR, "model2_classification.pt")
    if not os.path.exists(path): raise FileNotFoundError(path)
    m = tv_models.efficientnet_b0(weights=None)
    m.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.4, inplace=True),
        torch.nn.Linear(m.classifier[1].in_features, NUM_CLASSES),
    )
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    m.load_state_dict(state)
    return m.to(DEVICE).eval(), path

def _load_lstm():
    m3 = os.path.join(WEIGHTS_DIR, "model3_forecasting.pt")
    sc = os.path.join(WEIGHTS_DIR, "model3_scaler.pkl")
    m  = CycloneLSTM()
    state = torch.load(m3, map_location=DEVICE, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    m.load_state_dict(state); m.to(DEVICE).eval()
    with open(sc, "rb") as f: scaler = pickle.load(f)
    return m, scaler, m3

_M2_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ---- Model 1 ----------------------------------------------------------------
def eval_model1(args, results):
    _header("MODEL 1 - CYCLONE DETECTION  (YOLOv8-nano)")
    out = {"model": "YOLOv8-nano"}
    print("\n  Loading weights ...")
    t0 = time.perf_counter()
    try:
        yolo, path = _load_yolo()
        load_ms = (time.perf_counter()-t0)*1000
        print(f"  Weights  : {os.path.relpath(path, PROJECT_ROOT)}")
        print(f"  Load time: {load_ms:.1f} ms")
        out["load_time_ms"] = round(load_ms, 1)
    except Exception as e:
        print(f"  [FAIL] {e}"); results["model1"] = {"status":"FAIL","error":str(e)}; return

    _section("Live Inference (synthetic 640x640 image)")
    dummy = Image.new("RGB", (640,640), color=(10,20,40))
    t1 = time.perf_counter()
    res = yolo(dummy, verbose=False)[0]
    infer_ms = (time.perf_counter()-t1)*1000
    n = len(res.boxes) if res.boxes else 0
    print(f"  Inference : {infer_ms:.1f} ms  |  Detections: {n}  |  Status: {_ok(True)}")
    out.update({"inference_time_ms": round(infer_ms,1), "status":"PASS"})

    if args.m1_data:
        _section(f"Full mAP Evaluation  ->  {args.m1_data}")
        try:
            vr = yolo.val(data=args.m1_data, verbose=False)
            p,r,m50,m95 = float(vr.box.p.mean()),float(vr.box.r.mean()),float(vr.box.map50),float(vr.box.map)
            print(f"  Precision:{p:.3f}  Recall:{r:.3f}  mAP50:{m50:.3f}  mAP50-95:{m95:.3f}")
            out["live_scores"] = {"precision":round(p,3),"recall":round(r,3),"mAP50":round(m50,3),"mAP50_95":round(m95,3)}
        except Exception as e:
            print(f"  [FAIL] {e}"); out["eval_error"] = str(e)
    else:
        print("\n  TIP: Run with --m1-data path/to/data.yaml for real mAP scores.")
    results["model1"] = out

# ---- Model 2 ----------------------------------------------------------------
def eval_model2(args, results):
    _header("MODEL 2 - INTENSITY CLASSIFICATION  (EfficientNet-B0, 3-class)")
    out = {"model":"EfficientNet-B0","classes":CLASSES_M2}
    print("\n  Loading weights ...")
    t0 = time.perf_counter()
    try:
        effnet, path = _load_effnet()
        load_ms = (time.perf_counter()-t0)*1000
        print(f"  Weights  : {os.path.relpath(path, PROJECT_ROOT)}")
        print(f"  Classes  : {CLASSES_M2}   Device: {DEVICE}")
        print(f"  Load time: {load_ms:.1f} ms")
        out["load_time_ms"] = round(load_ms,1)
    except Exception as e:
        print(f"  [FAIL] {e}"); results["model2"] = {"status":"FAIL","error":str(e)}; return

    _section("Live Inference (synthetic 224x224 tensor)")
    dummy = torch.randn(1,3,224,224, device=DEVICE)
    t1 = time.perf_counter()
    with torch.no_grad():
        probs = torch.softmax(effnet(dummy), dim=1)[0].cpu().numpy()
    infer_ms = (time.perf_counter()-t1)*1000
    top = int(np.argmax(probs))
    prob_dict = {c: round(float(probs[i]),4) for i,c in enumerate(CLASSES_M2)}
    print(f"  Inference: {infer_ms:.1f} ms  |  Predicted: {CLASSES_M2[top]} ({probs[top]*100:.1f}%)  |  Status: {_ok(True)}")
    print(f"  All probs: {prob_dict}")
    out.update({"inference_time_ms":round(infer_ms,1),"status":"PASS","sample_probs":prob_dict})

    if args.m2_csv and args.m2_img_dir:
        _section("Full Classification Evaluation")
        print(f"  CSV : {args.m2_csv}\n  Imgs: {args.m2_img_dir}")
        try:
            import pandas as pd
            from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
            df = pd.read_csv(args.m2_csv)
            if "Category" in df.columns and "Category_merged" not in df.columns:
                df["Category_merged"] = df["Category"].replace({"SuCS":"VSCS","D":"D_DD","DD":"D_DD","SCS":"VSCS"})
            if "Set" in df.columns:
                df = df[df["Set"]=="Test"].reset_index(drop=True)
                print(f"  Test samples: {len(df)}")
            cat_to_idx = {c:i for i,c in enumerate(CLASSES_M2)}
            df = df[df["Category_merged"].isin(CLASSES_M2)].reset_index(drop=True)
            y_true, y_pred, errs = [], [], 0
            print(f"\n  Running inference on {len(df)} images ...")
            t_s = time.perf_counter()
            for idx, row in df.iterrows():
                try:
                    img = Image.open(os.path.join(args.m2_img_dir, row["filename"])).convert("RGB")
                    t   = _M2_TRANSFORM(img).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        pred = int(torch.softmax(effnet(t),dim=1).argmax())
                    y_true.append(cat_to_idx[row["Category_merged"]])
                    y_pred.append(pred)
                except: errs += 1
                if (idx+1) % 500 == 0:
                    print(f"    {idx+1}/{len(df)}  ({time.perf_counter()-t_s:.1f}s)")
            print(f"\n  Done: {len(y_true)} evaluated, {errs} skipped, {time.perf_counter()-t_s:.1f}s\n")
            acc = accuracy_score(y_true, y_pred)
            report = classification_report(y_true, y_pred, target_names=CLASSES_M2, digits=2)
            cm = confusion_matrix(y_true, y_pred).tolist()
            print(f"  Overall Accuracy: {_pct(acc)}\n")
            print(report)
            print(f"  Confusion Matrix (rows=true, cols=pred, classes={CLASSES_M2}):")
            for r in cm: print(f"    {r}")
            out["live_scores"] = {"overall_accuracy":round(acc,4),"report":report,"confusion_matrix":cm}
        except Exception as e:
            print(f"  [FAIL] {e}"); out["eval_error"] = str(e)
    else:
        print("\n  TIP: Run with --m2-csv labels.csv --m2-img-dir images/ for real metrics.")
    results["model2"] = out

# ---- Model 3 ----------------------------------------------------------------
def eval_model3(args, results):
    _header("MODEL 3 - TRACK & INTENSITY FORECASTING  (2-Layer LSTM)")
    out = {"model":"2-Layer LSTM"}
    print("\n  Loading weights ...")
    t0 = time.perf_counter()
    try:
        lstm, scaler, path = _load_lstm()
        load_ms = (time.perf_counter()-t0)*1000
        print(f"  Weights  : {os.path.relpath(path, PROJECT_ROOT)}   Device: {DEVICE}")
        print(f"  Load time: {load_ms:.1f} ms")
        out["load_time_ms"] = round(load_ms,1)
    except Exception as e:
        print(f"  [FAIL] {e}"); results["model3"] = {"status":"FAIL","error":str(e)}; return

    _section("Live Inference - 8-Step Recursive Forecast (sample Cyclone Fani track)")
    sample = np.array([[10.2+i*0.3, 80.1+i*0.4, 65.0+i*3.0] for i in range(8)], dtype=np.float32)
    t1 = time.perf_counter()
    curr = torch.tensor(scaler.transform(sample), dtype=torch.float32, device=DEVICE).unsqueeze(0)
    forecasts = []
    with torch.no_grad():
        for _ in range(8):
            pred_n = lstm(curr)
            forecasts.append(scaler.inverse_transform(pred_n.cpu().numpy())[0].tolist())
            curr = torch.cat([curr[:,1:,:], pred_n.unsqueeze(1)], dim=1)
    infer_ms = (time.perf_counter()-t1)*1000
    start = sample[-1]
    print(f"  Start: Lat {start[0]:.1f}N  Lon {start[1]:.1f}E  Wind {start[2]:.0f} km/h\n")
    horizons = ["+3h","+6h","+9h","+12h","+15h","+18h","+21h","+24h"]
    print(f"  {'Step':<5} {'Horizon':<8} {'Lat':>8} {'Lon':>8} {'Wind':>8}   Dist from start")
    print(f"  {'-'*65}")
    for i,(fc,h) in enumerate(zip(forecasts,horizons)):
        d = _haversine(start[0],start[1],fc[0],fc[1])
        print(f"  {i+1:<5} {h:<8} {fc[0]:>8.2f} {fc[1]:>8.2f} {fc[2]:>8.1f}   {d:.0f} km")
    print(f"\n  Inference: {infer_ms:.1f} ms (8 steps)  |  Status: {_ok(True)}")
    out.update({"inference_time_ms":round(infer_ms,1),"status":"PASS",
                "sample_forecast":[{"step":i+1,"horizon":h,"lat":round(fc[0],3),"lon":round(fc[1],3),"wind_kmh":round(fc[2],1)}
                                    for i,(fc,h) in enumerate(zip(forecasts,horizons))]})

    if args.m3_data:
        _section(f"Full Forecasting Evaluation  ->  {args.m3_data}")
        try:
            import pandas as pd
            df = pd.read_csv(args.m3_data)
            req = {"storm_id","lat","lon","wind_speed"}
            if not req.issubset(df.columns): raise ValueError(f"CSV needs columns: {req}")
            errs_km, errs_w = [], []
            for sid, grp in df.groupby("storm_id"):
                pts = grp[["lat","lon","wind_speed"]].values.astype(np.float32)
                if len(pts) < 9: continue
                for i in range(len(pts)-8):
                    seq, truth = pts[i:i+8], pts[i+8]
                    inp = torch.tensor(scaler.transform(seq), dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    with torch.no_grad():
                        pred = scaler.inverse_transform(lstm(inp).cpu().numpy())[0]
                    errs_km.append(_haversine(truth[0],truth[1],pred[0],pred[1]))
                    errs_w.append(abs(pred[2]-truth[2]))
            if errs_km:
                print(f"  Sequences: {len(errs_km)}")
                print(f"  Mean position error  : {np.mean(errs_km):.1f} km")
                print(f"  Median position error: {np.median(errs_km):.1f} km")
                print(f"  Mean wind error      : {np.mean(errs_w):.1f} km/h")
                out["live_scores"] = {"sequences":len(errs_km),"mean_pos_err_km":round(np.mean(errs_km),1),
                                      "median_pos_err_km":round(np.median(errs_km),1),"mean_wind_err_kmh":round(np.mean(errs_w),1)}
            else:
                print("  [WARN] No valid sequences found.")
        except Exception as e:
            print(f"  [FAIL] {e}"); out["eval_error"] = str(e)
    else:
        print("\n  TIP: Run with --m3-data ibtracs.csv for real forecasting errors.")
    results["model3"] = out

# ---- Summary ----------------------------------------------------------------
def _summary(results):
    print(); print(_bar()); print("  LIVE EVALUATION SUMMARY"); print(_bar())
    rows = [("model1","Model 1 - YOLO Detection"),("model2","Model 2 - EfficientNet"),("model3","Model 3 - LSTM")]
    print(f"  {'Model':<32} {'Load':>9} {'Infer':>9}  Status")
    print(f"  {'-'*32} {'-'*9} {'-'*9}  ------")
    for key, label in rows:
        r  = results.get(key,{})
        l  = f"{r['load_time_ms']} ms"  if "load_time_ms"  in r else "N/A"
        i  = f"{r['inference_time_ms']} ms" if "inference_time_ms" in r else "N/A"
        st = "\033[92mPASS\033[0m" if r.get("status")=="PASS" else "\033[91mFAIL\033[0m"
        print(f"  {label:<32} {l:>9} {i:>9}  {st}")
    print(_bar()); print()

# ---- Main -------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="CycloSense - Live Model Evaluation")
    p.add_argument("--model",      choices=["1","2","3","all"], default="all")
    p.add_argument("--m1-data",    default="", metavar="YAML")
    p.add_argument("--m2-csv",     default="", metavar="CSV")
    p.add_argument("--m2-img-dir", default="", metavar="DIR")
    p.add_argument("--m3-data",    default="", metavar="CSV")
    p.add_argument("--json",       action="store_true")
    p.add_argument("--export",     default="", metavar="FILE")
    args = p.parse_args()

    if not args.json:
        print(); print(_bar())
        print("       CYCLOSENSE - LIVE MODEL EVALUATION & INFERENCE BENCHMARK")
        print(_bar())
        print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  PyTorch  : {torch.__version__}  |  Device: {DEVICE}")
        print(f"  Weights  : {WEIGHTS_DIR}")
        print(_bar())

    results = {}
    if args.model in ("1","all"): eval_model1(args, results)
    if args.model in ("2","all"): eval_model2(args, results)
    if args.model in ("3","all"): eval_model3(args, results)
    if not args.json and args.model == "all": _summary(results)

    data = {"project":"CycloSense","timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "device":str(DEVICE),"results":results}
    if args.json:  print(json.dumps(data, indent=2))
    if args.export:
        with open(args.export,"w") as f: json.dump(data,f,indent=2)
        print(f"\n  Exported -> {os.path.abspath(args.export)}")

if __name__ == "__main__":
    main()
