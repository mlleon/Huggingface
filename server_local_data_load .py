import re

operation = input("请选择操作（下载/上传）：")

if operation == "下载" or operation == "上传":
    ssh_command = input("请输入SSH命令（示例：ssh -p 14835 root@connect.westb.seetacloud.com）：")

    # 使用正则表达式提取端口号和用户名主机名
    match = re.search(r"-p\s+(\d+).*?(\S+?@\S+)$", ssh_command)
    if match:
        ssh_port = match.group(1)
        ssh_user_host = match.group(2)
    else:
        ssh_port = "22"  # 如果没有指定端口号，默认为 22
        ssh_user_host = ""
        print("未能从SSH命令中提取到端口号和用户名主机名部分，请确保输入的SSH命令格式正确。")
        exit()

    server_path = input("请输入服务器文件/文件夹路径（示例：autodl-tmp/datasets/wikipedia）：")
    local_path = input("请输入本地文件/文件夹路径（示例：F:\\pythonProject\\datasets）：")

    if operation == "下载":
        scp_command = f"scp -rP {ssh_port} {ssh_user_host}:{server_path} {local_path}"
    else:
        scp_command = f"scp -rP {ssh_port} {local_path} {ssh_user_host}:{server_path}"

    print(scp_command)
else:
    print("无效的操作选择。请重新运行程序并输入有效操作。")
