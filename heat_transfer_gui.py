#!/usr/bin/env python3

import importlib
import importlib.util
import inspect
import sys
import traceback
import types
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from ast import literal_eval

APP_TITLE = "Heat Transfer – Solver GUI"
THIS_DIR = Path(__file__).resolve().parent

# Ensure local modules are importable
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

TARGET_MODULES = [
    ("Conduction", "conduction"),
    ("Free Convection", "free_convection"),
    ("Forced Convection", "forced_convection"),
    ("Radiation", "radiation"),
    ("Transient", "transient"),
    ("Fins", "fins"),
]

# Map common parameter symbols to their SI units so the GUI can display them.
# This is based on the notation used in the companion heat-transfer modules.
PARAM_UNITS = {
    "q": "W",                       # heat-transfer rate
    "q_total": "W",

    "k": "W/(m·K)",                 # thermal conductivity

    "h": "W/(m²·K)",                # convection / overall coefficients
    "h0": "W/(m²·K)",
    "h_c": "W/(m²·K)",
    "h_r": "W/(m²·K)",
    "h_total": "W/(m²·K)",

    "A": "m²",
    "A0": "m²",
    "A1": "m²",
    "A2": "m²",
    "A_base": "m²",
    "A_f": "m²",
    "A_fins": "m²",
    "A_s": "m²",

    "T": "K (or °C, consistent)",
    "T0": "K (or °C, consistent)",
    "T1": "K (or °C, consistent)",
    "T2": "K (or °C, consistent)",
    "T_s": "K (or °C, consistent)",
    "T_surf": "K (or °C, consistent)",
    "T_inf": "K (or °C, consistent)",
    "T_amb": "K (or °C, consistent)",

    # Dimensionless groups
    "Fo": "dimensionless",
    "Re": "dimensionless",
    "ReD": "dimensionless",
    "ReL": "dimensionless",
    "Pr": "dimensionless",
    "NuD": "dimensionless",
    "NuL": "dimensionless",
    "RaD": "dimensionless",
    "RaL": "dimensionless",

    # Material and fluid properties
    "Cp": "J/(kg·K)",
    "rho": "kg/m³",
    "mu": "Pa·s",
    "nu": "m²/s",
    "alpha": "m²/s",

    # Radiation
    "sigma": "W/(m²·K⁴)",
    "epsilon": "dimensionless",
    "eps1": "dimensionless",
    "eps2": "dimensionless",
    "eps_s": "dimensionless",
    "F12": "dimensionless",
    "F12_star": "dimensionless",
    "E": "W/m²",

    # Geometry / lengths
    "L": "m",
    "D": "m",
    "d": "m",
    "x": "m",
    "r1": "m",
    "r2": "m",
    "P": "m",
    "W": "m",

    # Other scalars
    "g": "m/s²",
    "m": "1/m",         # fin parameter
    "v": "m/s",
    "t": "s",
    "n": "dimensionless",
    "n_terms": "dimensionless",
    "eta_f": "dimensionless",
}


def get_units(symbol_name):
    """
    Return a human-readable units string for a given parameter name, or None
    if no units are known. This lets the GUI show labels like "q [W]" and
    "T1 [K (or °C, consistent)]".
    """
    if not symbol_name:
        return None

    unit = PARAM_UNITS.get(symbol_name)
    if unit is not None:
        return unit

    # Light-weight heuristics so that new functions keep getting reasonable units
    # without having to update this table every time.
    if symbol_name.startswith("T"):
        return "K (or °C, consistent)"
    if symbol_name.startswith("A"):
        return "m²"
    if symbol_name.startswith("h"):
        return "W/(m²·K)"
    if symbol_name in {"L", "D", "d", "x", "r1", "r2", "P", "W"}:
        return "m"
    if symbol_name in {"Fo", "Re", "ReD", "ReL", "Pr", "NuD", "NuL", "RaD", "RaL"}:
        return "dimensionless"

    return None

def try_import(module_name: str):
    """
    Try to import a module by name from THIS_DIR. Returns (module | None, error_text | None).
    """
    try:
        spec = importlib.util.spec_from_file_location(module_name, THIS_DIR / "src" / f"{module_name}.py")
        if spec is None or spec.loader is None:
            return None, f"Could not find {module_name}.py next to this GUI."
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod, None
    except Exception:
        return None, traceback.format_exc()

def find_functions(mod: types.ModuleType):
    """
    Return a list of (name, function, signature, doc_first_line) for callables defined in `mod`.
    Only includes functions defined in the module (not imports).
    """
    out = []
    for name, obj in inspect.getmembers(mod, inspect.isfunction):
        if getattr(obj, "__module__", "") == mod.__name__:
            sig = inspect.signature(obj)
            doc = (inspect.getdoc(obj) or "").strip().splitlines()
            doc_first = doc[0] if doc else ""
            out.append((name, obj, sig, doc_first))
    # Sort by name for stable display
    out.sort(key=lambda t: t[0].lower())
    return out

def coerce_value(text: str):
    """
    Convert string input to a Python value.
    - Empty string => None
    - Try literal_eval for numbers/tuples/booleans/etc.
    - Fallback to raw string.
    """
    if text.strip() == "":
        return None
    try:
        return literal_eval(text)
    except Exception:
        # try float as a convenience
        try:
            return float(text)
        except Exception:
            return text

class FunctionRunner(ttk.Frame):
    def __init__(self, parent, module_title: str, module_name: str):
        super().__init__(parent)
        self.module_title = module_title
        self.module_name = module_name
        self.mod = None
        self.import_error = None
        self.functions = []  # list of (name, func, sig, doc_first)
        self.param_vars = {}  # name -> tk.StringVar

        self._build_ui()
        self.reload_module(initial=True)

    def _build_ui(self):
        # Top row: function chooser + reload
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=(10, 6))

        ttk.Label(top, text=f"{self.module_title} function:").pack(side="left")
        self.func_combo = ttk.Combobox(top, state="readonly", values=[])
        self.func_combo.pack(side="left", padx=6, fill="x", expand=True)
        self.func_combo.bind("<<ComboboxSelected>>", self._on_func_selected)

        self.reload_btn = ttk.Button(top, text="Reload Modules", command=self._on_reload_clicked)
        self.reload_btn.pack(side="right")

        # Middle: dynamic parameter form
        self.form = ttk.LabelFrame(self, text="Inputs (leave exactly one blank to solve for it if the function supports None)")
        self.form.pack(fill="both", expand=True, padx=10, pady=6)

        # Bottom: compute + output
        bottom = ttk.Frame(self)
        bottom.pack(fill="both", expand=False, padx=10, pady=(6, 10))

        self.compute_btn = ttk.Button(bottom, text="Compute", command=self._on_compute)
        self.compute_btn.pack(side="left")

        self.output = tk.Text(self, height=10, wrap="word")
        self.output.pack(fill="both", expand=True, padx=10, pady=(6, 10))

    def reload_module(self, initial=False):
        self.mod, self.import_error = try_import(self.module_name)

        if self.import_error:
            self.func_combo.configure(values=[])
            self.func_combo.set("")
            self.compute_btn.state(["disabled"])
            self._show_import_error(self.import_error)
            return

        # Discover functions
        self.functions = find_functions(self.mod)
        names = [name for (name, *_rest) in self.functions]
        self.func_combo.configure(values=names)
        if names:
            self.compute_btn.state(["!disabled"])
            # Select first function on initial load
            if initial:
                self.func_combo.current(0)
                self._on_func_selected()
        else:
            self.compute_btn.state(["disabled"])
            self._write_output(f"No functions found in {self.module_name}.py.")

    def _clear_form(self):
        for child in self.form.winfo_children():
            child.destroy()
        self.param_vars.clear()

    def _on_func_selected(self, *_evt):
        self._clear_form()
        func_name = self.func_combo.get()
        if not func_name:
            return
        # Fetch signature
        fn = None
        for name, func, sig, _doc in self.functions:
            if name == func_name:
                fn = (func, sig)
                break
        if fn is None:
            return
        _, sig = fn

        # Create an input row per parameter
        row = 0
        for pname, param in sig.parameters.items():
            # Only support POSITIONAL_OR_KEYWORD and KEYWORD params
            if param.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                continue

            label_text = pname
            unit = get_units(pname)
            if unit:
                label_text = f"{pname} [{unit}]"

            ttk.Label(self.form, text=label_text).grid(row=row, column=0, sticky="w", padx=8, pady=4)
            var = tk.StringVar()
            self.param_vars[pname] = var

            entry = ttk.Entry(self.form, textvariable=var, width=28)
            entry.grid(row=row, column=1, sticky="ew", padx=8, pady=4)

            # Show default if present
            default = param.default
            if default is not inspect._empty:
                if default is None:
                    hint = "(default: None)"
                else:
                    hint = f"(default: {default})"
                ttk.Label(self.form, text=hint, foreground="#666").grid(row=row, column=2, sticky="w", padx=8)
            row += 1

        # Add a small hint about types
        ttk.Label(self.form, text="Tip: leave blank for None; you can also enter tuples like (1.2, 3.4).",
                  foreground="#666").grid(row=row, column=0, columnspan=3, sticky="w", padx=8, pady=(8, 0))

    def _on_reload_clicked(self):
        # Reload all target modules to pick up code changes
        for _title, modname in TARGET_MODULES:
            if modname in sys.modules:
                try:
                    importlib.reload(sys.modules[modname])
                except Exception:
                    # Ignore errors here; they'll be surfaced by this tab's reload
                    pass
        self.reload_module(initial=False)
        messagebox.showinfo("Reloaded", "Modules reloaded.")

    def _write_output(self, text: str):
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("end", text)
        self.output.configure(state="disabled")

    def _show_import_error(self, err_text: str):
        self._write_output(
            f"Could not import '{self.module_name}.py'.\n\n"
            "Details:\n"
            f"{err_text}\n\n"
            "Fix the error in the module and click 'Reload Modules'."
        )

    def _on_compute(self):
        if self.import_error:
            self._show_import_error(self.import_error)
            return

        func_name = self.func_combo.get()
        if not func_name:
            messagebox.showwarning("Select a function", "Please choose a function first.")
            return

        # Get function object and signature
        fn = None
        for name, func, sig, _doc in self.functions:
            if name == func_name:
                fn = (func, sig)
                break
        if fn is None:
            self._write_output("Internal error: function not found.")
            return

        func, sig = fn

        # Build kwargs from form entries
        kwargs = {}
        for pname, var in self.param_vars.items():
            txt = var.get()
            val = coerce_value(txt)
            if val is None and sig.parameters[pname].default is not inspect._empty:
                # leave as missing to use default
                continue
            kwargs[pname] = val

        try:
            result = func(**kwargs)

            # 1) Prefer a label provided by the function itself
            label = getattr(func, "last_solved", None)

            # 2) Fallback: infer from blank UI fields (old behaviour)
            if not label:
                solved = [
                    name for name, var in self.param_vars.items()
                    if var.get().strip() == ""
                ]
                if len(solved) == 1:
                    label = solved[0]
                elif solved:
                    label = ", ".join(solved)

            # Pretty print result
            self._write_output(self._format_result(result, solved_label=label))
        except Exception as e:
            tb = traceback.format_exc()
            self._write_output(f"Error calling {func.__name__}:\n{tb}")

def _format_result(self, result, solved_label=None):
        header = "Result"
        if solved_label:
            header += f" for '{solved_label}'"
            # If we know the physical units for this symbol, append them.
            # We only attempt this for a single, simple symbol name.
            unit = get_units(str(solved_label).strip())
            if unit:
                header += f" [{unit}]"
        if isinstance(result, tuple):
            lines = [header + ":"]
            for i, item in enumerate(result, 1):
                lines.append(f"  [{i}] {item!r}")
            return "\n".join(lines)
        else:
            return f"{header}: {result!r}"



def main():
    root = tk.Tk()
    root.title(APP_TITLE)

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    for title, modname in TARGET_MODULES:
        frame = FunctionRunner(notebook, title, modname)
        notebook.add(frame, text=title)

    # Minimum decent size
    root.geometry("820x640")
    root.mainloop()

if __name__ == "__main__":
    main()