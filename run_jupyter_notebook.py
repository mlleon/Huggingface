import subprocess
import psutil
import time


def start_jupyter():
    # 检查 Jupyter 是否已经在运行
    for process in psutil.process_iter():
        if "jupyter-notebook" in process.name():
            print("Jupyter Notebook 已经在运行中。")
            return

    # 启动 Jupyter Notebook 服务
    try:
        subprocess.Popen(["jupyter", "notebook"], stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate(
            timeout=10)
        print("Jupyter Notebook 服务已经启动。")
        # 等待一段时间以确保服务已经启动
        time.sleep(3)
    except Exception as e:
        print(f"启动 Jupyter Notebook 服务时出现错误：{e}")


def stop_jupyter():
    # 停止 Jupyter Notebook 服务
    for process in psutil.process_iter():
        if "jupyter-notebook" in process.name():
            try:
                process.terminate()
                print("Jupyter Notebook 服务已经停止。")
                return
            except Exception as e:
                print(f"停止 Jupyter Notebook 服务时出现错误：{e}")
    print("Jupyter Notebook 未在运行中。")


if __name__ == "__main__":
    while True:
        try:
            choice = input(
                "请选择要执行的操作：\n1. 启动 Jupyter Notebook 服务\n2. 停止 Jupyter Notebook 服务\n选择：")
            if choice == "1":
                start_jupyter()
            elif choice == "2":
                stop_jupyter()
            elif choice == "3":
                break
            else:
                print("无效的选项，请重新输入。")
        except KeyboardInterrupt:
            print("\n操作已被中断。")
            break
