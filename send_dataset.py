import serial
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 1. 配置物理串口参数 (请与设备管理器中的实际端口一致)
SERIAL_PORT = 'COM8'
BAUD_RATE = 9600

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    print(f"成功建立通信链路: {SERIAL_PORT} @ {BAUD_RATE}")
    time.sleep(2)  # 等待单片机复位稳定
except Exception as e:
    print(f"通信链路建立失败, 请检查端口是否被串口调试助手占用: {e}")
    exit()

# 2. 加载数据集并划分验证集 (保持与训练阶段同源的随机种子)
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

print(f"验证数据集加载完毕，共计含有 {len(X_test)} 组测试样本。准备开始流式传输...")
print("==================================================================")

# 3. 流式传输及响应读取
for i in range(len(X_test)):
    features = X_test[i]
    label = y_test[i]

    # 按照单片机约定的 CSV 字符串协议打包数据格式
    data_string = f"{features[0]},{features[1]},{features[2]},{features[3]},{label}\n"

    # 执行串口硬件发送
    ser.write(data_string.encode('utf-8'))

    # 阻塞等待单片机的实时推理计算反馈输出
    time.sleep(0.1)  # 给硬件层留出基础的数据处理窗长

    while ser.in_waiting > 0:
        response = ser.readline().decode('utf-8', errors='ignore')
        print(response.strip())

ser.close()
print("==================================================================")
print("自动化测试集验证流程执行完毕。通信链路已安全断开。")