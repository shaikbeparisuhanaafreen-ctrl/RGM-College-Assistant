

import platform
import sys
import os
import importlib

print("=" * 60)
print(" 🎓 RGMCET College AI Assistant")
print(" 🔍 Environment & Dependency Check")
print("=" * 60)


print(f"\n🐍 Python Version      : {sys.version.split()[0]}")
print(f"🖥️  OS                 : {platform.system()} {platform.release()}")
print(f"⚙️  Architecture       : {platform.machine()}")


libraries = [
    "torch",
    "sentence_transformers",
    "faiss",
    "flask",
    "transformers",
    "numpy",
    "pandas"
]

print("\n📦 Library Status:\n")

for lib in libraries:
    try:
        module = importlib.import_module(lib)
        version = getattr(module, "__version__", "Unknown")
        print(f"{lib:<25} ✅ Installed (version: {version})")
    except ImportError:
        print(f"{lib:<25} ❌ Not Installed")



print("\n✅ Environment check completed.")
print("=" * 60)