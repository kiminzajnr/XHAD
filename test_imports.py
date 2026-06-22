print("\n" + "="*60)
print("TESTING ALL LIBRARY IMPORTS")
print("="*60 + "\n")

libraries = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'scikit-learn': 'sklearn',
    'tensorflow': 'tensorflow',
    'keras': 'keras',
    'shap': 'shap',
    'scipy': 'scipy',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'statsmodels': 'statsmodels',
    'jupyter': 'jupyter',
    'drain3': 'drain3'
}

failed = []
for name, module in libraries.items():
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✅ {name:20s} version: {version}")
    except Exception as e:
        print(f"❌ {name:20s} FAILED: {e}")
        failed.append(name)

print("\n" + "="*60)
if not failed:
    print("✅ ALL LIBRARIES LOADED SUCCESSFULLY!")
else:
    print(f"❌ {len(failed)} libraries failed: {failed}")
print("="*60 + "\n")
