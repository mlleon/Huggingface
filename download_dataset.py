import subprocess
import os
from datasets import load_dataset

dataset_name = "wikipedia"  # Change this to the desired dataset name if needed
subset_name = "20220301.simple"  # Change this to the desired subset name if needed

# 1、使用AutoDL学术加速
result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True,
                        text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value

# 2、使用 load_dataset 函数加载感兴趣的数据集
dataset = load_dataset(dataset_name, subset_name)


# 3、创建一个用于保存数据集的本地目录
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


datasets_directory = "/root/autodl-tmp/datasets"
create_directory(datasets_directory)
subset_path = os.path.join(datasets_directory, dataset_name, subset_name) if subset_name else os.path.join(
    datasets_directory, dataset_name)
create_directory(subset_path)

# 4、将数据集文件保存到本地目录
dataset.save_to_disk(os.path.abspath(subset_path)) if not os.listdir(subset_path) else None
