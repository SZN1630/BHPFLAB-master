import subprocess
import sys

# 读取 run.sh 文件中的命令
with open('run.sh', 'r') as f:
    commands = [line.strip() for line in f if line.strip()]

# 依次执行每个命令
for i, cmd in enumerate(commands, 1):
    print(f'执行命令 {i}/{len(commands)}: {cmd}')
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f'命令 {i} 执行失败，退出码: {result.returncode}')
        sys.exit(result.returncode)

print('所有命令执行完成')
