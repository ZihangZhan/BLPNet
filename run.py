import subprocess

# 定义要运行的次数
num_runs = 3

# 循环运行 main.py
for _ in range(num_runs):
    subprocess.run(["python", "main.py"])
