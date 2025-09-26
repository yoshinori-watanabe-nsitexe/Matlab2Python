from __future__ import annotations
import pandas as pd
import os
from typing import Any, Dict
import numpy as np
import scipy.io as sio
import h5py

unique=lambda x:list(set(x))
#matlab functions
table =lambda:pd.DataFrame()
array2table=lambda x:pd.DataFrame(x)
table2array=lambda x: np.array(x)
strcat= lambda aa:''.join(aa)
fullfile=lambda *args:os.path.join(*args)
rot90=lambda mat, k=1:np.rot90(mat, k)
writetable=lambda df, file_path: pd.DataFrame(df).to_csv(file_path, index=False)
writematrix= lambda m,name:np.save(name,m)

def intersect(a, b):
    a1, ia = np.unique(a, return_index=True)
    b1, ib = np.unique(b, return_index=True)
    aux = np.concatenate((a1, b1))
    aux.sort()
    c = aux[:-1][aux[1:] == aux[:-1]]
    return c, ia[np.isin(a1, c)], ib[np.isin(b1, c)]

## === 正規化（訓練統計のみでフィッティング→リーク防止）===
def normalize(Train_raw,Val_raw=None,dim=0):
#    print("Train_raw",Train_raw.shape)
    muX    = np.mean(Train_raw, dim)
    sigmaX = np.std(Train_raw, dim)
    sigmaX=sigmaX+1e-7

    Train = (Train_raw - muX) / sigmaX
    if(Val_raw is None):
        return Train,None,muX,sigmaX
    else:
        Val   = (Val_raw   - muX) / sigmaX
        return Train,Val,muX,sigmaX

def height(x):
    return x.shape[0]

def downsample(x, factor):
    return x[::factor]

def strrep(str, old, new):
    return str.replace(old, new)

def loadmat_unified(path: str,
                    simplify: bool = True,
                    prefer_mat73: bool = True,
                    v73_fallback: str = "warn"  # "warn" | "error" | "minimal"
                    ) -> Dict[str, Any]:
    """
    MATLAB .mat を v7.2以前 / v7.3 を自動判別して読み込み、Python ネイティブに寄せて返す。

    Parameters
    ----------
    path : str
        .mat ファイルパス
    simplify : bool
        可能なら cell→list、サイズ1次元の圧縮などを有効化
    prefer_mat73 : bool
        v7.3 のとき mat73 を優先して使用（推奨）
    v73_fallback : {"warn","error","minimal"}
        mat73 が無い/失敗した場合の挙動
        - "warn": 警告を出して最小限の h5py リーダで返す（数値・文字列・cell中心）
        - "error": 例外を送出
        - "minimal": 警告なしで最小限リーダ

    Returns
    -------
    Dict[str, Any]
        {変数名: 値} の辞書（内部メタ変数 __header__ 等は除外）
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    is_v73 = False
    try:
        is_v73 = h5py.is_hdf5(path)
    except Exception:
        # 失敗時は v7.2以前として扱う
        is_v73 = False

    if is_v73:
        # --- v7.3 (HDF5) ---
        if prefer_mat73:
            try:
                import mat73  # type: ignore
                data = mat73.loadmat(path)
                return _strip_private(data)
            except Exception as e:
                if v73_fallback == "error":
                    raise RuntimeError(
                        "v7.3 の読み込みに mat73 を使えませんでした。`pip install mat73` を検討してください。"
                    ) from e
                if v73_fallback == "warn":
                    print("[loadmat_unified] mat73 による v7.3 読み込みに失敗。h5py の最小限フォールバックに切り替えます。")
        # 最小限フォールバック
        return _load_v73_minimal_with_h5py(path)

    else:
        # --- v7.2 以前 ---
        kwargs = dict(struct_as_record=False, squeeze_me=True)
        # SciPy 1.10+ のときだけ simplify_cells を渡す
        if simplify:
            try:
                # 存在しないキーワードを渡すと TypeError なのでここで確認
                import inspect
                if "simplify_cells" in inspect.signature(sio.loadmat).parameters:
                    kwargs["simplify_cells"] = True  # type: ignore
            except Exception:
                pass

        raw = sio.loadmat(path, **kwargs)
        raw = _strip_private(raw)
        # struct/cell を可能な範囲で Python ネイティブに再帰変換
        return {k: _to_native_from_scipy(v, simplify=simplify) for k, v in raw.items()}


# ---------- helpers ----------

def _strip_private(d: Dict[str, Any]) -> Dict[str, Any]:
    """__header__ など内部キーを除く"""
    return {k: v for k, v in d.items() if not k.startswith("__")}

def _to_native_from_scipy(obj: Any, simplify: bool = True) -> Any:
    """
    SciPy loadmat(v7.2以前) の戻り値に含まれる
    - mat_struct（構造体）
    - object配列（cell配列）
    をできるだけ Python ネイティブに再帰変換。
    """
    # numpy 配列（object/数値/論理/文字列）
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            # cell配列など
            if obj.ndim == 0:
                return _to_native_from_scipy(obj.item(), simplify=simplify)
            # 1D/2D などの object 配列 → list(list(...)) に展開
            return [[_to_native_from_scipy(x, simplify=simplify) for x in row] for row in obj.tolist()] \
                   if obj.ndim > 1 else [_to_native_from_scipy(x, simplify=simplify) for x in obj.tolist()]
        else:
            # 数値/論理/文字列の nparray はそのまま
            return obj

    # SciPy の mat_struct（構造体）検出
    # バージョンにより型パスが異なる場合があるので属性で判定
    if hasattr(obj, "_fieldnames"):
        out = {}
        for name in obj._fieldnames:  # type: ignore
            val = getattr(obj, name)
            out[name] = _to_native_from_scipy(val, simplify=simplify)
        return out

    # Python の list/tuple も再帰的に処理
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_native_from_scipy(x, simplify=simplify) for x in obj)

    # スカラ（np.generic含む）
    if isinstance(obj, (np.generic,)):
        return np.asarray(obj).item()

    # それ以外はそのまま
    return obj


def _load_v73_minimal_with_h5py(path: str) -> Dict[str, Any]:
    """
    v7.3（HDF5）を h5py だけで“最小限”読む。
    - 数値/論理/文字列/cell(=参照の配列) を扱う
    - 構造体（struct）は mat73 の利用を推奨（ここでは未完全）
    """
    out: Dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            out[k] = _read_h5obj(f[k])
    return out

def _read_h5obj(obj: h5py.Dataset | h5py.Group) -> Any:
    # Dataset（配列 or 文字列）
    if isinstance(obj, h5py.Dataset):
        # 文字列は asstr() でデコード（h5py >=3）
        if obj.dtype.kind in ("S", "O", "U"):
            try:
                return obj.asstr()[()]
            except Exception:
                data = obj[()]
                # bytes -> str
                if isinstance(data, (bytes, np.bytes_)):
                    return data.decode("utf-8", errors="replace")
                return data
        else:
            return obj[()]

    # Group（cell/struct などが入る）
    if isinstance(obj, h5py.Group):
        # MATLAB は cell を参照（オブジェクト参照）の配列で保持することが多い
        # cell 配列の可能性（'MATLAB_class'='cell'）を先に見る
        cls = obj.attrs.get("MATLAB_class", None)
        if isinstance(cls, (bytes, np.bytes_)):
            cls = cls.decode("utf-8", errors="ignore")

        if cls == "cell":
            # 要素は Dataset of references
            if "data" in obj:
                refs = obj["data"][()]
                # スカラー cell
                if np.ndim(refs) == 0:
                    return _read_h5ref(obj.file, refs)
                # 配列 cell -> list(…) に展開
                return [_read_h5ref(obj.file, r) for r in refs.ravel()].__iter__() and \
                       np.array([_read_h5ref(obj.file, r) for r in refs.ravel()], dtype=object).reshape(refs.shape).tolist()
            # 形が違う場合は中のキーを辿る
            return {kk: _read_h5obj(obj[kk]) for kk in obj.keys()}

        if cls == "struct":
            # 最小限：フィールドごとに読み出し（ネスト/配列structなどの完全対応は mat73 推奨）
            # 多くの v7.3 struct は "fieldnames" と "data" を持つ
            try:
                fieldnames = [s.decode("utf-8") if isinstance(s, (bytes, np.bytes_)) else str(s)
                              for s in obj["fieldnames"][()]]
                data = obj["data"]
                # 単一 struct（0次元 or 1x1）を想定してフィールドを取り出す
                result = {}
                for i, fn in enumerate(fieldnames):
                    ref = data[:, i][()] if data.ndim == 2 else data[i]
                    # ref が参照であることを想定
                    val = _read_h5ref(obj.file, ref[0] if np.ndim(ref) else ref)
                    result[fn] = val
                return result
            except Exception:
                # 形が合わないものは生の中身を辞書で返す
                return {kk: _read_h5obj(obj[kk]) for kk in obj.keys()}

        # それ以外の group は中のキーを辞書化
        return {kk: _read_h5obj(obj[kk]) for kk in obj.keys()}

    # ここには来ない想定
    return obj

def _read_h5ref(f: h5py.File, ref) -> Any:
    """オブジェクト参照から中身を読む"""
    try:
        target = f[ref]
        return _read_h5obj(target)
    except Exception:
        return None

