from itertools import islice

file= 'scripts/results/SingleControl/1\\heading\\ppo\\v1\\render\\v1.txt.acmi'
# 打开 ACMI 文件进行读取
if __name__ == '__main__':
    with open(file, 'r') as file:
        for line in islice(file,3,None):
            # 如果行以 # 开头，跳过注释行
            if line.startswith("#"):
                continue

            # 按逗号分隔字段
            fields = line.split(',')

            # 获取速度字段所在的部分（例如 T=119.99999999999999|59.999999999999986|5575.672055298579|35.52659889123547|-7.156248033292914e-15|22.629172440451807）
            data_part = fields[1].split('|')

            # 提取第四列数据（35.52659889123547）
            roll = data_part[3]
            roll_file = 'D:/HCH/LAG/log/roll_list.txt'
            with open(roll_file, 'a') as file:
                file.write("{}\n".format(roll))
            # 输出提取的速度数据
            print(roll)