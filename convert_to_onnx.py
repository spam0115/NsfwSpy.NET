#!/usr/bin/env python
"""
Convert a HuggingFace safetensors model to ONNX format.

Usage:
    python convert_to_onnx.py <input_model_dir> [output_dir]

Arguments:
    input_model_dir   Directory containing the HuggingFace model files
                      (must include config.json and model.safetensors)
    output_dir        Optional. Directory to save the ONNX model.
                      Defaults to ./NsfwSpy/ (the project's model directory)

Options:
    -h, --help        Show this help message

Examples:
    python convert_to_onnx.py ./NsfwSpy/newmodel
    python convert_to_onnx.py ./my_model ./custom_output

The script automatically installs required Python packages if not present.
Supported architectures include image classification, causal LM, sequence
classification, token classification, masked LM, and question answering models.
"""

import importlib
import json
import os
import subprocess
import sys


REQUIRED_PACKAGES = [
    ("optimum", "optimum[onnxruntime]"),
    ("transformers", "transformers"),
    ("torch", "torch"),
    ("safetensors", "safetensors"),
    ("onnx", "onnx"),
]


def show_help():
    print(__doc__.strip())


def ensure_packages():
    installed_any = False
    for module_name, pip_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(module_name)
        except ImportError:
            print(f"Installing {pip_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            importlib.invalidate_caches()
            installed_any = True
    if installed_any:
        for module_name, _ in REQUIRED_PACKAGES:
            importlib.import_module(module_name)


def get_ort_model_class(architectures):
    """Determine the ORTModel class based on the model's architecture names."""
    arch_str = " ".join(architectures)

    if "ImageClassification" in arch_str:
        from optimum.onnxruntime import ORTModelForImageClassification
        return ORTModelForImageClassification, "image classification"
    elif "ForCausalLM" in arch_str:
        from optimum.onnxruntime import ORTModelForCausalLM
        return ORTModelForCausalLM, "causal LM"
    elif "ForSequenceClassification" in arch_str:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        return ORTModelForSequenceClassification, "sequence classification"
    elif "ForTokenClassification" in arch_str:
        from optimum.onnxruntime import ORTModelForTokenClassification
        return ORTModelForTokenClassification, "token classification"
    elif "ForMaskedLM" in arch_str:
        from optimum.onnxruntime import ORTModelForMaskedLM
        return ORTModelForMaskedLM, "masked LM"
    elif "ForQuestionAnswering" in arch_str:
        from optimum.onnxruntime import ORTModelForQuestionAnswering
        return ORTModelForQuestionAnswering, "question answering"
    else:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        return ORTModelForFeatureExtraction, "feature extraction (fallback)"


def main():
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        show_help()
        sys.exit(0 if args and args[0] in ("-h", "--help") else 1)

    if len(args) > 2:
        print(f"Error: too many arguments (expected 1-2, got {len(args)})\n")
        show_help()
        sys.exit(1)

    input_dir = os.path.abspath(args[0])

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output = os.path.join(script_dir, "NsfwSpy")
    output_dir = os.path.abspath(args[1]) if len(args) > 1 else default_output

    if not os.path.isdir(input_dir):
        print(f"Error: input directory does not exist: {input_dir}")
        sys.exit(1)

    config_path = os.path.join(input_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"Error: no config.json found in: {input_dir}")
        print("The input directory must contain config.json and model.safetensors.")
        sys.exit(1)

    safetensors_path = os.path.join(input_dir, "model.safetensors")
    if not os.path.exists(safetensors_path):
        print(f"Error: no model.safetensors found in: {input_dir}")
        sys.exit(1)

    print("Checking Python dependencies...")
    ensure_packages()
    print("All dependencies satisfied.\n")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    architectures = config.get("architectures", [])
    if not architectures:
        print("Warning: no 'architectures' field in config.json, using fallback (feature extraction).")
        architectures = ["Unknown"]

    model_type = config.get("model_type", "unknown")
    model_class, task_name = get_ort_model_class(architectures)

    print("Model details:")
    print(f"  Input directory:  {input_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Architecture:     {', '.join(architectures)}")
    print(f"  Model type:       {model_type}")
    print(f"  Detected task:    {task_name}\n")

    print("Exporting to ONNX (this may take a minute)...")
    model = model_class.from_pretrained(input_dir, export=True)

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)

    onnx_path = os.path.join(output_dir, "model.onnx")
    if os.path.exists(onnx_path):
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"Done! ONNX model saved: {onnx_path}")
        print(f"  File size: {size_mb:.1f} MB")
    else:
        print(f"Done! ONNX model saved to: {output_dir}")

    print("\nNext steps:")
    print("  1. Ensure the .onnx file is referenced in your .csproj:")
    print('     <None Update="model.onnx"><CopyToOutputDirectory>Always</CopyToOutputDirectory></None>')
    print("  2. Load it in C# with: new InferenceSession(modelPath)")


if __name__ == "__main__":
    main()
