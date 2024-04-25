import subprocess
import webbrowser
import time

DOC_CANNO_EXECUTABLE = 'doccano'
DOC_CANNO_PORT = '8000'
DOC_CANNO_URL = f'http://127.0.0.1:{DOC_CANNO_PORT}/'

DOC_CANNO_DIR = 'D:\\Anaconda\\envs\\text_annotae\\Scripts'


def start_doccano():
    try:
        # 在指定目录下启动 Doccano webserver
        subprocess.Popen([DOC_CANNO_EXECUTABLE, 'webserver', '--port', DOC_CANNO_PORT],
                         cwd=DOC_CANNO_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Doccano web 服务已启动！")
        # 打开默认浏览器访问 Doccano
        webbrowser.open(DOC_CANNO_URL)

        # 后台运行 doccano task
        subprocess.Popen([DOC_CANNO_EXECUTABLE, 'task'],
                         cwd=DOC_CANNO_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         startupinfo=subprocess.STARTUPINFO(wShowWindow=subprocess.SW_HIDE))
        print("Doccano task 已启动！")
    except FileNotFoundError:
        print(f"未找到 {DOC_CANNO_EXECUTABLE} 可执行文件，请确保已安装 Doccano 并正确配置路径。")
    except Exception as e:
        print(f"启动 Doccano 时出错：{e}")


if __name__ == "__main__":
    print("\n选择操作:")
    print("1. 启动 Doccano")

    choice = input("请输入选项号码: ")

    if choice == "1":
        start_doccano()

    # 给用户一点时间来查看输出
    time.sleep(2)

