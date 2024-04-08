import subprocess
import psutil
import webbrowser
import time

DOC_CANNO_EXECUTABLE = 'doccano'
DOC_CANNO_PORT = '8000'
DOC_CANNO_URL = f'http://127.0.0.1:{DOC_CANNO_PORT}/'


def start_doccano():
    try:
        # 启动 Doccano 进程
        subprocess.Popen([DOC_CANNO_EXECUTABLE, 'webserver', '--port', DOC_CANNO_PORT],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Doccano web 服务已启动！")
        # 打开默认浏览器访问 Doccano
        webbrowser.open(DOC_CANNO_URL)
    except FileNotFoundError:
        print(f"未找到 {DOC_CANNO_EXECUTABLE} 可执行文件，请确保已安装 Doccano 并正确配置路径。")
    except Exception as e:
        print(f"启动 Doccano 时出错：{e}")


def stop_doccano():
    try:
        # 查找 Doccano 进程并终止它
        for proc in psutil.process_iter():
            if "python" in proc.name() and DOC_CANNO_EXECUTABLE in proc.cmdline():
                proc.kill()
                print("Doccano 已关闭！")
                return
        print("Doccano 进程未找到。")
    except Exception as e:
        print(f"关闭 Doccano 时出错：{e}")


if __name__ == "__main__":
    while True:
        print("\n选择操作:")
        print("1. 启动 Doccano")
        print("2. 关闭 Doccano")

        choice = input("请输入选项号码: ")

        if choice == "1":
            start_doccano()
        elif choice == "2":
            stop_doccano()
        elif choice == "3":
            print("退出程序。")
            break
        else:
            print("无效的选项。")

    # 给用户一点时间来查看输出
    time.sleep(2)
