from __future__ import annotations
import argparse
import re
import sys
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# --- Function mapping ----------------------------------------------------------

FUNC_MAP_CALL_SIMPLE = {
    # basic math
    'abs': 'np.abs',
    'sqrt': 'np.sqrt',
    'exp': 'np.exp',
    'log': 'np.log',
    'log10': 'np.log10',
    'sin': 'np.sin',
    'cos': 'np.cos',
    'tan': 'np.tan',
    'asin': 'np.arcsin',
    'acos': 'np.arccos',
    'atan': 'np.arctan',
    'atan2': 'np.arctan2',
    'sinh': 'np.sinh',
    'cosh': 'np.cosh',
    'tanh': 'np.tanh',
    'round': 'np.round',
    'floor': 'np.floor',
    'ceil': 'np.ceil',
    'mod': 'np.mod',
    'rem': 'np.remainder',
    'sum': 'np.sum',
    'mean': 'np.mean',
    'median': 'np.median',
    'std': 'np.std',
    'var': 'np.var',
    'min': 'np.min',
    'max': 'np.max',
    'cumsum': 'np.cumsum',
    'cumprod': 'np.cumprod',
    'diff': 'np.diff',

    # array creation
    'zeros': 'np.zeros',
    'ones': 'np.ones',
    'eye': 'np.eye',
    'diag': 'np.diag',
    'linspace': 'np.linspace',
    'meshgrid': 'np.meshgrid',
    #numpy
    "save":"np.save",
    "load":"np.load",

    # random
    'rand': 'np.random.rand',
    'randn': 'np.random.randn',
    'rng':'np.manual_seed',

    # plotting (matplotlib)
    'plot': 'plt.plot',
    'semilogx': 'plt.semilogx',
    'semilogy': 'plt.semilogy',
    'loglog': 'plt.loglog',
    'scatter': 'plt.scatter',
    'hist': 'plt.hist',
    'imshow': 'plt.imshow',
    'imagesc': 'plt.imshow',
    'surf': 'plt.imshow',  # crude
    'title': 'plt.title',
    'xlabel': 'plt.xlabel',
    'ylabel': 'plt.ylabel',
    'legend': 'plt.legend',
    'subplot': 'plt.subplot',
    'figure': 'plt.figure',
    'hold': '',  # ignored
    'grid': 'plt.grid',
    'xlim': 'plt.xlim',
    'ylim': 'plt.ylim',
    'axis': 'plt.axis',
    "hold on":"",
    #os
    "mkdir":"os.makedirs",
    #utils
    "hold on":"",
    "hold on":"",
    "hold on":"",
    "hold on":"",

    #timer
    "tic":"tic=time.time()",
    "toc":"elapsedtime=time.time()-tic",
}

BLOCK_START = ("if", "for", "while", "switch", "try", "function")
BLOCK_MID   = ("elseif", "else", "case", "otherwise", "catch")
BLOCK_END   = ("end",)

header=["""
# translated from MATLAB (heuristic)
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from scipy import io
from scipy.signal import find_peaks
import glob
import plot_validation_result_2 as p2
from torch import Tensor,nn,optim,numel,randperm,manual_seed
import pandas as pd
import json
from train_utils import create_TrainLoader,create_TestLoader,train_model,evaluate_model
import datetime
import time
import os
import utils        
"""]


def postprocess(s: str) -> str:
    s=re.sub(r'^hold on', '', s)
    s=re.sub(r'^clc', '', s)
    s=re.sub(r'^close all', '', s)
    s=re.sub(r'^clearvars', '', s)
    s=re.sub(r'find_peaks\(([^() ]+)\)',r'find_peaks(\1,prominence=5)', s)
    s=s.replace("true","True")
    s=s.replace("false","False")

    #配列のみ() to [] にしたい
    f=re.match(r'([^() ]+)\(',s)
    #if(not f in FUNC_MAP_CALL_SIMPLE.keys() ): #pre
    if(not f in list(FUNC_MAP_CALL_SIMPLE.values()) ): #post
        s=re.sub(r'([^() ]+)\((.+)\)', r'\1[\2]', s)

    return s

def strip_trailing_semicolon(s: str) -> str:
    # remove trailing semicolons (not in strings)
    # naive: drop ending ';' possibly with spaces
    return re.sub(r';\s*$', '', s)

#return re.sub(r'^%',"#",line)

def replace_1line_comments(line: str) -> str:
    return re.sub(r'^%',"#",line)

def replace_comments(line: str) -> str:
    return re.sub(r'%',"#",line)
    
def _replace_comments(line: str) -> str:
    # MATLAB: % comment  /  %{ ... %} block comments are handled elsewhere
    # Avoid % inside strings: simple heuristic by splitting on quotes
    parts = re.split(r"('.*?')", line)  # keep quoted segments
    for i in range(0, len(parts), 2):
        if '%' in parts[i]:
            parts[i] = parts[i].split('%', 1)[0] + '#' + parts[i].split('%', 1)[1] if False else parts[i].split('%', 1)[0]
    # A simpler approach: cut at first % outside quotes
    # Rebuild quickly by scanning
    out = []
    in_str = False
    for ch in line:
        if ch == "'" and not in_str:
            in_str = True
            out.append(ch)
        elif ch == "'" and in_str:
            in_str = False
            out.append(ch)
        elif ch == '%' and not in_str:
            # turn the rest into comment
            out.append('#')
            break
        else:
            out.append(ch)
    else:
        return ''.join(out)
    # append nothing further (drop trailing original)
    return ''.join(out)

def split_ellipsis(text: str) -> str:
    # Join lines with MATLAB line continuation '...'
    lines = text.splitlines()
    joined = []
    buf = ""
    for ln in lines:
        if re.search(r"\.\.\.\s*$", ln.strip()):
            buf += re.sub(r"\.\.\.\s*$", " ", ln.strip())
        else:
            if buf:
                joined.append(buf + ln.strip())
                buf = ""
            else:
                joined.append(ln)
    if buf:
        joined.append(buf)
    return "\n".join(joined)    

# --- Converter core ------------------------------------------------------------
@dataclass
class Block:
    kind: str  # function/if/for/while/switch/try
    data: Optional[dict] = field(default_factory=dict)

class MatlabToPythonConverter:
    def __init__(self, assume_matmul: bool = True):
        self.indent = 0
        self.blocks: List[Block] = []
        self.used_numpy = False
        self.used_matplotlib = False
        self.assume_matmul = assume_matmul
        self.lines_out: List[str] = []

    # --- high-level API ---
    def convert(self, text: str) -> Tuple[str, List[str]]:
        """Return (python_code, warnings)"""
        text = split_ellipsis(text)
        lines = text.splitlines()
        warnings: List[str] = []

        # Insert header later, after scanning usage
        body_lines: List[str] = []
        for raw in lines:
            conv, ws = self.convert_line(raw.rstrip("\n"))
            warnings.extend(ws)
            if conv is None:
                continue
            if isinstance(conv, list):
                body_lines.extend(conv)
            else:
                body_lines.append(conv)

        # Close any remaining blocks gracefully
        while self.blocks:
            blk = self.blocks.pop()
            self.indent = max(self.indent - 1, 0)
            if blk.kind == "function":
                outs = blk.data.get("outs", [])
                if outs:
                    body_lines.append(" " * (self.indent*4) + "return " + ", ".join(outs))

        # Prepend imports if used
        return "".join(header + body_lines), warnings

    # --- line conversion ---
    def convert_line(self, line: str) -> Tuple[Optional[str|List[str]], List[str]]:
        line = line.rstrip()
        # drop empty
        if not line.strip():
            return ("\n", [])
        # block comments
        if re.match(r'^\s*%{\s*$', line):
            return (self.ind() + '"""', [])
        if re.match(r'^\s*%}\s*$', line):
            return ('"""' + "\n", [])

        # Replace inline comments
        line = replace_1line_comments(line)

        line = replace_comments(line)

        # Strip trailing ';'
        line = strip_trailing_semicolon(line)

        # Trim leading spaces for parsing
        stripped = line.strip()

        # Function header
        m = re.match(r'^function\s*(\[(.*?)\]|([^\s=]+))\s*=\s*([A-Za-z_]\w*)\s*\((.*?)\)\s*$', stripped)
        if m:
            outs_block, outs_inner, single_out, fname, args = m.groups()
            outs = []
            if outs_inner is not None:
                outs = [o.strip() for o in outs_inner.split(',') if o.strip()]
            elif single_out is not None:
                outs = [single_out.strip()]
            args = args.strip()
            if args == "": args_py = ""
            else: args_py = args
            # open function block
            hdr = self.ind() + f"def {fname}({args_py}):\n"
            self.blocks.append(Block("function", {"outs": outs}))
            self.indent += 1
            inits = []
            for o in outs:
                inits.append(self.ind() + f"{o} = None\n")
            return ([hdr] + inits, [])

        # Function header (no outputs): function fname(args)
        m = re.match(r'^function\s+([A-Za-z_]\w*)\s*\((.*?)\)\s*$', stripped)
        if m:
            fname, args = m.groups()
            hdr = self.ind() + f"def {fname}({args}):\n"
            self.blocks.append(Block("function", {"outs": []}))
            self.indent += 1
            return (hdr, [])

        # elseif/else/if/for/while/switch/try/case/otherwise/catch
        # elseif
        if re.match(r'^\s*elseif\b', stripped):
            cond = stripped[len('elseif'):].strip()
            cond = self._convert_condition(cond)
            # decrease indent then elif
            self.indent = max(self.indent - 1, 0)
            return (self.ind() + f"elif {cond}:\n", [])

        if re.match(r'^\s*else\b', stripped):
            # decrease indent then else
            self.indent = max(self.indent - 1, 0)
            return (self.ind() + "else:\n", [])

        if re.match(r'^\s*if\b', stripped):
            cond = stripped[len('if'):].strip()
            cond = self._convert_condition(cond)
            self.blocks.append(Block("if"))
            out = self.ind() + f"if {cond}:\n"
            self.indent += 1
            return (out, [])

        if re.match(r'^\s*for\b', stripped):
            rhs = stripped[len('for'):].strip()
            m = re.match(r'^([A-Za-z_]\w*)\s*=\s*(.+)$', rhs)
            if m:
                var, rng = m.groups()
                rng_py = self._convert_for_range(rng.strip())
                self.blocks.append(Block("for"))
                out = self.ind() + f"for {var} in {rng_py}:\n"
                self.indent += 1
                return (out, [])
            else:
                # fallback
                self.blocks.append(Block("for"))
                out = self.ind() + "# TODO: for-loop not parsed\n" + self.ind() + f"# {stripped}\n"
                self.indent += 1
                return (out, ["for-loop not parsed: " + stripped])

        if re.match(r'^\s*while\b', stripped):
            cond = stripped[len('while'):].strip()
            cond = self._convert_condition(cond)
            self.blocks.append(Block("while"))
            out = self.ind() + f"while {cond}:\n"
            self.indent += 1
            return (out, [])

        if re.match(r'^\s*try\s*$', stripped):
            self.blocks.append(Block("try"))
            out = self.ind() + "try:\n"
            self.indent += 1
            return (out, [])

        if re.match(r'^\s*catch\b', stripped):
            # decrease indent, open except
            self.indent = max(self.indent - 1, 0)
            self.blocks.append(Block("catch"))
            out = self.ind() + "except Exception as e:\n"
            self.indent += 1
            return (out, [])

        if re.match(r'^\s*switch\b', stripped):
            expr = stripped[len('switch'):].strip()
            self.blocks.append(Block("switch", {"expr": expr, "first": True}))
            # no direct switch in Python; we'll emit a comment
            return (self.ind() + f"# switch {expr}\n", [])

        if re.match(r'^\s*case\b', stripped):
            val = stripped[len('case'):].strip()
            # lookup last switch
            for blk in reversed(self.blocks):
                if blk.kind == "switch":
                    first = blk.data.get("first", True)
                    blk.data["first"] = False
                    expr = blk.data.get("expr", "EXPR")
                    if first:
                        # open if
                        self.blocks.append(Block("if"))
                        out = self.ind() + f"if {expr} == {val}:\n"
                        self.indent += 1
                        return (out, [])
                    else:
                        # change to elif (drop one indent first)
                        self.indent = max(self.indent - 1, 0)
                        out = self.ind() + f"elif {expr} == {val}:\n"
                        self.indent += 1
                        return (out, [])
            return (self.ind() + f"# case {val}\n", [])

        if re.match(r'^\s*otherwise\b', stripped):
            self.indent = max(self.indent - 1, 0)
            out = self.ind() + "else:\n"
            self.indent += 1
            return (out, [])

        if re.match(r'^\s*end\s*$', stripped):
            # close block
            if self.blocks:
                blk = self.blocks.pop()
                self.indent = max(self.indent - 1, 0)
                if blk.kind == "function":
                    outs = blk.data.get("outs", [])
                    if outs:
                        return (self.ind() + "return " + ", ".join(outs) + "\n", [])
                    return ("\n", [])
                return ("\n", [])
            else:
                return ("\n", [])

        # disp → print
        if re.match(r'^\s*disp\s*\(', stripped):
            content = stripped[stripped.find('('):]
            return (self.ind() + "print" + content + "\n", [])

        # fprintf → print (naive)
        if re.match(r'^\s*fprintf\s*\(', stripped):
            content = stripped[stripped.find('('):]
            return (self.ind() + "print" + content + "\n", [])

        # Regular line: operators & calls
        py = stripped

        # Operator replacements
        # element-wise first
        py = re.sub(r'\.\^', '**', py)
        py = re.sub(r'\.\*', '*', py)
        py = re.sub(r'\./', '/', py)
        # power
        py = re.sub(r'(?<!\.)\^', '**', py)
        # logicals
        py = re.sub(r'~=', '!=', py)
        py = re.sub(r'&&', ' and ', py)
        py = re.sub(r'\|\|', ' or ', py)
        # unary not: ~A -> ~A in Python is bitwise not; better to map to not
        py = re.sub(r'~(?=\w|\()', ' not ', py)
        # pi
        py = re.sub(r'\bpi\b', 'np.pi', py)

        # transpose A' -> A.T (rough; skip if within quotes or followed by ')')
        #py = re.sub(r"\b([A-Za-z_]\w*)\s*'", r"\1.T", py)

        # size(A,1) → A.shape[0], size(A) → A.shape
        py = re.sub(r'\bsize\(\s*([A-Za-z_]\w*)\s*,\s*1\s*\)', r'\1.shape[0]', py)
        py = re.sub(r'\bsize\(\s*([A-Za-z_]\w*)\s*,\s*2\s*\)', r'\1.shape[1]', py)
        py = re.sub(r'\bsize\(\s*([A-Za-z_]\w*)\s*\)', r'\1.shape', py)
        py = re.sub(r'\blength\(\s*([A-Za-z_]\w*)\s*\)', r'len(\1)', py)
        py = re.sub(r'\bnumel\(\s*([A-Za-z_]\w*)\s*\)', r'\1.size', py)
        py = re.sub(r'\bassignin\(s*([A-Za-z_]\w*),([A-Za-z_]\w*)r,([A-Za-z_]\w*)\s*\)', r'\1\[\2\]=\3', py)
        # replace function calls (simple map), track numpy/matplotlib usage
        def _replace_calls(m):
            name = m.group(1)
            if name in ("size","length","numel"):  # already handled
                return m.group(0)
            mapped = FUNC_MAP_CALL_SIMPLE.get(name)
            if mapped is None:
                return m.group(0)
            if mapped.startswith("np."):
                self.used_numpy = True
            if mapped.startswith("plt."):
                self.used_matplotlib = True
            return mapped + "(" + m.group(2) + ")"
        py = re.sub(r'\b([A-Za-z_]\w*)\s*\(\s*([^\)]*)\s*\)', _replace_calls, py)

        # rand/zeros shapes: MATLAB uses zeros(m,n) → np.zeros((m,n))
        py = re.sub(r'\bnp\.zeros\(\s*([^\)]*?)\s*\)', r'np.zeros((\1))', py)
        py = re.sub(r'\bnp\.ones\(\s*([^\)]*?)\s*\)', r'np.ones((\1))', py)

        # anonymous function @(x,y) expr → lambda x,y: expr
        py = re.sub(r'@\( *([^\)]*?) *\)\s*', r'lambda \1: ', py)

        # matrix multiply: * → @ (heuristic)
        if self.assume_matmul:
            # avoid changing comparisons etc.; change plain * to @ except when obvious scalar? heuristic only
            # crude: if there is ' * ' not surrounded by digits-only; still risky.
            py = re.sub(r'(?<!\.)\*(?!\*)', '@', py)

        py=postprocess(py)
        # comment already converted; ensure trailing newline
        out = self.ind() + py + "\n"

        # mark numpy usage if we saw 'np.' or operators like '**' that suggest numpy? (not strictly needed)
        if re.search(r'\bnp\.', out):
            self.used_numpy = True

        return (out, [])

    def _convert_condition(self, cond: str) -> str:
        cond = cond.strip()
        if cond.startswith('(') and cond.endswith(')'):
            cond = cond[1:-1].strip()
        cond = re.sub(r'~=', '!=', cond)
        cond = re.sub(r'&&', ' and ', cond)
        cond = re.sub(r'\|\|', ' or ', cond)
        cond = re.sub(r'~(?=\w|\()', ' not ', cond)
        return cond

    def _convert_for_range(self, rng: str) -> str:
        # Handle a:b or a:s:b
        rng = rng.strip()
        # Common 1:N
        m = re.match(r'^(\w+)\s*:\s*(\w+)$', rng)
        if m:
            a, b = m.groups()
            # MATLAB inclusive; Python range is exclusive → +1
            return f"range({a}, {b}+1)"
        # With step: a:s:b
        m = re.match(r'^(\w+)\s*:\s*(\w+)\s*:\s*(\w+)$', rng)
        if m:
            a, s, b = m.groups()
            # need to overshoot end depending on sign
            return f"range({a}, {b}+({s}), {s})"
        # Fallback: try length(x) etc.
        rng = rng.replace("length(", "len(")
        return rng  # as-is

    def ind(self) -> str:
        return " " * (self.indent * 4)

# --- Public API ---------------------------------------------------------------
def convert_str(matlab_text: str, assume_matmul: bool = True) -> Tuple[str, List[str]]:
    return MatlabToPythonConverter(assume_matmul=assume_matmul).convert(matlab_text)

def convert_file(in_path: str, out_path: Optional[str] = None, assume_matmul: bool = True) -> Tuple[str, List[str]]:
    with open(in_path, 'r', encoding='utf-8') as f:
        src = f.read()
    py, warns = convert_str(src, assume_matmul=assume_matmul)
    if out_path:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(py)
    return py, warns

# --- CLI ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Heuristic MATLAB(.m) → Python(Numpy/Matplotlib) converter")
    ap.add_argument("input", help="MATLAB .m file")
    ap.add_argument("-o", "--output", help="output .py file")
    ap.add_argument("--no-matmul", action="store_true", help="do NOT convert '*' to '@'")
    args = ap.parse_args()
    py, warns = convert_file(args.input, args.output, assume_matmul=not args.no_matmul)
    if warns:
        sys.stderr.write("\n".join(f"[warn] {w}" for w in warns) + "\n")
    if not args.output:
        sys.stdout.write(py)

if __name__ == "__main__":
    main()
