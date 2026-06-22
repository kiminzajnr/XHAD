import os

print("\n" + "="*60)
print("TESTING DATA LOADING")
print("="*60 + "\n")

# Path to data
data_path = "../data/"

# Test 1: Check files exist
print("1. CHECKING DATA FILES...")
files = {
    'HDFS_sample.log': os.path.join(data_path, 'HDFS_sample.log'),
    'BGL_sample.log': os.path.join(data_path, 'BGL_sample.log'),
}

for name, path in files.items():
    if os.path.exists(path):
        size_kb = os.path.getsize(path) / 1024
        print(f"   ✅ {name:25s} ({size_kb:.1f} KB)")
    else:
        print(f"   ❌ {name:25s} NOT FOUND")

# Test 2: Load HDFS logs
print("\n2. LOADING HDFS LOGS...")
try:
    with open(data_path + 'HDFS_sample.log', 'r') as f:
        hdfs_lines = f.readlines()
    print(f"   ✅ Loaded {len(hdfs_lines)} HDFS log lines")
    print(f"   Sample log 1: {hdfs_lines[0][:70]}...")
    print(f"   Sample log 2: {hdfs_lines[1][:70]}...")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: Load BGL logs
print("\n3. LOADING BGL LOGS...")
try:
    with open(data_path + 'BGL_sample.log', 'r') as f:
        bgl_lines = f.readlines()
    print(f"   ✅ Loaded {len(bgl_lines)} BGL log lines")
    print(f"   Sample log 1: {bgl_lines[0][:70]}...")
    print(f"   Sample log 2: {bgl_lines[1][:70]}...")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "="*60)
print("✅ DATA LOADING TEST COMPLETE")
print("="*60 + "\n")
